def score(points):
    r = np.linalg.norm(points, axis=1)
    boundary_min = 1.0 - np.max(r)
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists) / 2.0
    return min(boundary_min, min_dist)
