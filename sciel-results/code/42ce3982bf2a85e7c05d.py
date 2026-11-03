    def _sa_energy(circles, w, h, lambda_rad=50.0):
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        
        # Boundary violations
        pen_left = np.maximum(0, rs - xs)
        pen_right = np.maximum(0, xs + rs - w)
        pen_bottom = np.maximum(0, rs - ys)
        pen_top = np.maximum(0, ys + rs - h)
        
        boundary_violation = np.sum(pen_left**2) + np.sum(pen_right**2) + np.sum(pen_bottom**2) + np.sum(pen_top**2)
        
        # Pairwise overlaps
        # Vectorized pairwise distance
        dx = xs[:, np.newaxis] - xs[np.newaxis, :]
        dy = ys[:, np.newaxis] - ys[np.newaxis, :]
        d = np.sqrt(dx**2 + dy**2 + 1e-12)
        
        # Sum of radii pairs
        r_sum = rs[:, np.newaxis] + rs[np.newaxis, :]
        
        overlap = np.maximum(0, r_sum - d)
        # Exclude diagonal
        np.fill_diagonal(overlap, 0)
        
        overlap_violation = np.sum(overlap**2)
        
        total_violation = boundary_violation + overlap_violation
        radius_reward = lambda_rad * np.sum(rs)
        
        return total_violation - radius_reward
