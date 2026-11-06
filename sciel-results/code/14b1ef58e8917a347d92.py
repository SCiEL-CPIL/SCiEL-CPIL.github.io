def construct():
    # Two clusters of 6 consecutive integers, spaced far apart.
    # Cluster 1: 0-5, Cluster 2: 40-45
    # Within each cluster of 6, max Sidon subset is 2.
    # With Sidon-like spacing between clusters, cross sums won't create larger Sidon sets.
    return [0, 1, 2, 3, 4, 5, 40, 41, 42, 43, 44, 45]
