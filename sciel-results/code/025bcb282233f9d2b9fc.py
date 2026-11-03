<<<<<<< SEARCH
    def objective(z):
        # Unflatten
        z = z.reshape(N, 3)
        x = z[:, 0]
        y = z[:, 1]
        r = z[:, 2]
        
        # Sum of radii (we minimize negative sum)
        sum_r = np.sum(r)
        
        # Overlap penalty
        # Calculate pairwise distances
        dist = cdist(np.column_stack((x, y)), np.column_stack((x, y)))
        # We need upper triangle
        iu = np.triu_indices(N, k=1)
        d_ij = dist[iu]
        r_i = r[iu[0]]
        r_j = r[iu[1]]
        overlaps = np.maximum(0, r_i + r_j - d_ij)
        penalty_overlap = np.sum(overlaps**2)
        
        # Boundary penalty
        # r_i <= x_i, r_i <= 1-x_i, etc.
        pen_b = 0.0
        pen_b += np.sum(np.maximum(0, r - x)**2)
        pen_b += np.sum(np.maximum(0, r - (1 - x))**2)
        pen_b += np.sum(np.maximum(0, r - y)**2)
        pen_b += np.sum(np.maximum(0, r - (1 - y))**2)
        
        # Total penalty
        penalty = penalty_overlap + pen_b
        
        return -sum_r + 1000.0 * penalty  # Increased penalty weight to enforce validity
=======
    def objective(z):
        return objective(z)
>>>>>>> REPLACE

