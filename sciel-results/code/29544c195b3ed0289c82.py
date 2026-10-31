    def voronoi_radii_vec(P):
        n = len(P)
        # Distances to boundaries
        dist_b = np.min(np.stack([P[:, 0], 1 - P[:, 0], P[:, 1], 1 - P[:, 1]]), axis=1)
        
        # Distances to other points
        # d[i, j] = ||P[i] - P[j]||
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        # Min distance to neighbor / 2
        dist_n = np.min(d, axis=1) / 2.0
        
        return np.minimum(dist_b, dist_n)

    def voronoi_grad_vec(P):
        n = len(P)
        grad = np.zeros_like(P)
        
        # Compute radii and active constraints
        r = voronoi_radii_vec(P)
        
        # Distances to boundaries
        dist_b_x0 = P[:, 0]
        dist_b_x1 = 1 - P[:, 0]
        dist_b_y0 = P[:, 1]
        dist_b_y1 = 1 - P[:, 1]
        dist_b = np.stack([dist_b_x0, dist_b_x1, dist_b_y0, dist_b_y1], axis=1)
        active_b_idx = np.argmin(dist_b, axis=1)
        
        # Distances to neighbors
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        dist_n = d / 2.0
        active_n_idx = np.argmin(dist_n, axis=1)
        
        # Determine which constraint is active for each point
        min_b = np.min(dist_b, axis=1)
        min_n = np.min(dist_n, axis=1)
        
        for i in range(n):
            if min_b[i] < min_n[i]:
                # Boundary active
                idx = active_b_idx[i]
                if idx == 0: grad[i, 0] += 1.0
                elif idx == 1: grad[i, 0] -= 1.0
                elif idx == 2: grad[i, 1] += 1.0
                elif idx == 3: grad[i, 1] -= 1.0
            else:
                # Neighbor active
                j = active_n_idx[i]
                v = (P[i] - P[j]) / d[i, j]
                grad[i, 0] += 0.5 * v[0]
                grad[i, 1] += 0.5 * v[1]
                grad[j, 0] -= 0.5 * v[0]
                grad[j, 1] -= 0.5 * v[1]
                
        return grad
