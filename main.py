# DBSCAN(DB, dist, eps, minPts) {
#    C = 0                                              /* Cluster counter */
#    for each point P in database DB {
#       if label(P) ≠ undefined then continue           /* Previously processed in inner loop */
#       Neighbors N = RangeQuery(DB, dist, P, eps)      /* Find neighbors */
#       if |N| < minPts then {                          /* Density check */
#          label(P) = Noise                             /* Label as Noise */
#          continue
#       }
#       C = C + 1                                       /* next cluster label */
#       label(P) = C                                    /* Label initial point */
#       Seed set S = N \ {P}                            /* Neighbors to expand */
#       for each point Q in S {                         /* Process every seed point */
#          if label(Q) = Noise then label(Q) = C        /* Change Noise to border point */
#          if label(Q) ≠ undefined then continue        /* Previously processed */
#          label(Q) = C                                 /* Label neighbor */
#          Neighbors N = RangeQuery(DB, dist, Q, eps)   /* Find neighbors */
#          if |N| ≥ minPts then {                       /* Density check */
#             S = S ∪ N                                 /* Add new neighbors to seed set */
#          }
#       }
#    }
# }
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import datasets

import concurrent.futures


def region_query(P, eps, D):
    neighbour_pts = {}
    for index, point in enumerate(D):
        if np.sqrt(np.sum(np.square(P - point[0, :2]))) < eps:
            neighbour_pts[index] = point

    return neighbour_pts


def dbscan(X, eps, min_pts):
    matrix = np.matrix(X)
    target = np.full((X.shape[0], 1), -1)
    data = np.concatenate((matrix, target), axis=1)
    c = 0
    for index, row in enumerate(data):
        if row[0, -1] == -1:
            neighbours = region_query(row[0, :2], eps, data)
            if len(neighbours.keys()) < min_pts:
                data[index, -1] = -2
            else:
                data[index, -1] = c
                while len(neighbours.keys()) > 0:
                    point_key = [k for k in neighbours.keys()][0]
                    point = neighbours.pop(point_key)
                    if point[0, -1] == -2:
                        data[point_key, -1] = c
                    if point[0, -1] == -1:
                        data[point_key, -1] = c
                        new_neighbours = region_query(point[0, :2], eps, data)
                        if len(new_neighbours):
                            neighbours.update(new_neighbours)
                c += 1
                # data1 = pd.DataFrame(data=data)
                # g = sns.FacetGrid(data1, hue=2, palette="Set1", size=5)
                # g.map(plt.scatter, 0, 1, s=100, linewidth=.5, edgecolor="white")
                # g.add_legend()
                # plt.show()

    data1 = pd.DataFrame(data=data)
    g = sns.FacetGrid(data1, hue=2, palette="Set1", size=5)
    g.map(plt.scatter, 0, 1, s=100, linewidth=.5, edgecolor="white")
    g.add_legend()
    plt.show()


if __name__ == '__main__':
    n_samples = 500
    random_state = 170
    transformation = [[0.6, -0.6], [-0.4, 0.8]]

    models = [
        {
            'X': datasets.make_blobs(n_samples=n_samples, random_state=8, center_box=(-2000, 2000), cluster_std=50)[0],
            'eps': 100,
            'min_pts': 10,
        },
        {
            'X': datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)[0],
            'eps': 0.2,
            'min_pts': 6,
        },
        {
            'X': datasets.make_moons(n_samples=n_samples, noise=.05)[0],
            'eps': 0.2,
            'min_pts': 6,
        },
        {
            'X': datasets.make_blobs(n_samples=n_samples, random_state=8)[0],
            'eps': 0.5,
            'min_pts': 15,
        },
        {
            'X': np.random.rand(n_samples, 2),
            'eps': 0.3,
            'min_pts': 6,
        },
        {
            # Anisotropicly distributed data
            'X': np.dot(datasets.make_blobs(n_samples=n_samples, random_state=random_state)[0], transformation),
            'eps': 0.4,
            'min_pts': 6,
        },
        {
            # blobs with varied variances
            'X': datasets.make_blobs(n_samples=n_samples,
                                     cluster_std=[1.0, 2.5, 0.5],
                                     random_state=random_state)[0],
            'eps': 0.3,
            'min_pts': 12,
        },
        {
            'X': datasets.make_blobs(n_samples=n_samples, n_features=2, centers=5)[0],
            'eps': 0.9,
            'min_pts': 15,
        },
        {
            'X': datasets.make_gaussian_quantiles(n_samples=n_samples, n_features=2, n_classes=6)[0],
            'eps': 0.9,
            'min_pts': 8,
        },
        {
            'X': datasets.make_circles(n_samples=n_samples, factor=0.5)[0],
            'eps': 0.25,
            'min_pts': 5,
        },
        {
            'X': datasets.make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=1,
                                              n_clusters_per_class=1)[0],
            'eps': 0.3,
            'min_pts': 8,
        },
        {
            'X': datasets.make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2,
                                              n_clusters_per_class=1)[0],
            'eps': 0.3,
            'min_pts': 12,
        },
        {
            'X': datasets.make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2)[0],
            'eps': 0.5,
            'min_pts': 8,
        },
        {
            'X': datasets.make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2,
                                              n_clusters_per_class=1, n_classes=3)[0],
            'eps': 0.1,
            'min_pts': 6,
        },
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as e:
        for m in models:
            e.submit(dbscan, **m)
