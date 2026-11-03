    def calc_radius(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        norms = np.sqrt(np.sum(pts ** 2, axis=1))
        min_bound_dist = np.min(1.0 - norms)
        
        return min(min_pair_dist / 2.0, min_bound_dist)
