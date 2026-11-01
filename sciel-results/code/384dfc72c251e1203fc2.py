    import numpy as np
    from scipy.optimize import minimize

    def construct():
        N = 50
        
        def calc_radius(centers):
            # centers shape (N, 3)
            # Pairwise
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            # Mask diagonal
            mask = ~np.eye(N, dtype=bool)
            min_pair = np.min(dist[mask])
            
            # Walls
            abs_c = np.abs(centers)
            max_coord = np.max(abs_c, axis=1)
            wall_dist = 0.5 - max_coord
            min_wall = np.min(wall_dist)
            
            return min(min_pair / 2.0, min_wall)

        def objective(x):
            c = x.reshape(N, 3)
            # Clip to cube to ensure validity during optimization?
            # If we clip, the gradient becomes discontinuous.
            # Better to use a penalty or just let it go outside and rely on the wall term in radius?
            # If it goes outside, wall_dist becomes negative, radius becomes negative.
            # Minimizing -radius will push it back in.
            r = calc_radius(c)
            return -r

        # Initialization 1: Perturbed 4x4x4 Grid
        ticks = np.array([-0.375, -0.125, 0.125, 0.375])
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        # We need 50. Grid has 64.
        # Randomly select 50 indices.
        # To be deterministic in "best" search, we might want to try a few selections.
        # But for speed, let's pick the 50 that are "most central" or just random with fixed seed.
        # Actually, removing corner points is usually good.
        # Corner points in 4x4x4 grid (relative to center):
        # The grid is symmetric.
        # Let's just take a random subset with a fixed seed for reproducibility, then optimize.
        rng = np.random.default_rng(42)
        idx = rng.choice(64, 50, replace=False)
        x0 = grid[idx].copy()
        # Add small noise
        x0 += rng.normal(0, 0.01, x0.shape)
        
        res1 = minimize(objective, x0, method='Powell', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        best_c = res1.x.reshape(N, 3)
        best_r = -res1.fun

        # Initialization 2: Different random subset
        idx2 = rng.choice(64, 50, replace=False)
        x02 = grid[idx2].copy()
        x02 += rng.normal(0, 0.01, x02.shape)
        
        res2 = minimize(objective, x02, method='Powell', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        if -res2.fun > best_r:
            best_c = res2.x.reshape(N, 3)
            best_r = -res2.fun
            
        return best_c
    