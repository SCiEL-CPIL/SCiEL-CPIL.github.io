    def calculate_min_dist(pts):
        min_d = 1.0
        # Boundary
        dists_to_boundary = 0.5 - np.max(np.abs(pts), axis=1)
        min_d = min(min_d, np.min(dists_to_boundary))
        
        # Pairwise
        # diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        # dists = np.sqrt(np.sum(diff**2, axis=-1))
        # np.fill_diagonal(dists, np.inf)
        # min_d = min(min_d, np.min(dists))
        
        # This creates a 50x50 array. 2500 elements. Very fast.
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_d = min(min_d, np.min(dists))
        
        return min_d
