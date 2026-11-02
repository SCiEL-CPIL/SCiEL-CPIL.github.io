import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
        t_start = time.time()
        deadline = t_start + 140.0
        
        # Initialization
        m = 7
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        # grid shape (49, 2)
        
        # Remove center (index 24 in 7x7? 3*7+3 = 24)
        # Center is at (0,0).
        # Find index of (0,0)
        center_idx = np.argmin(np.sum(grid**2, axis=1))
        grid_no_center = np.delete(grid, center_idx, axis=0)
        
        # Add 2 points
        new_pts = np.array([[0, 0.06], [0, -0.06]])
        x0 = np.vstack([grid_no_center, new_pts])
        
        # Scale to ensure inside and slightly smaller to avoid initial overlaps
        x0 *= 0.95
        
        # Shuffle to break symmetry?
        # x0 = x0[np.random.permutation(N)] # Maybe keep grid structure
        
        def objective(x_flat):
            x = x_flat.reshape(N, 2)
            # Pairwise distances
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            # Mask diagonal
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
            
            # Boundary distances
            # x is in [-0.5, 0.5]
            # dist to boundary = min(0.5 - x, 0.5 + x) for each coord?
            # No, dist to square boundary.
            # dist_x = 0.5 - np.abs(x[:, 0])
            # dist_y = 0.5 - np.abs(x[:, 1])
            # min boundary dist = min(dist_x, dist_y) for each point, then global min?
            # The radius is limited by the closest boundary for any circle.
            # r <= min_i (min(dist_x_i, dist_y_i))
            
            dist_x = 0.5 - np.abs(x[:, 0])
            dist_y = 0.5 - np.abs(x[:, 1])
            boundary_dists = np.minimum(dist_x, dist_y)
            min_bound_dist = np.min(boundary_dists)
            
            # Combine
            # We want max(min(min_pair_dist, min_bound_dist))
            # Soft min of the set {min_pair_dist, min_bound_dist}
            # Or soft min of all individual distances?
            # Using all individual distances is more robust.
            
            all_dists = np.concatenate([dists.flatten(), boundary_dists])
            # Filter out inf
            all_dists = all_dists[np.isfinite(all_dists)]
            
            if len(all_dists) == 0:
                return 0.0
                
            # Soft min
            beta = 100.0 # Adaptive?
            # To avoid overflow, shift by min
            d_min = np.min(all_dists)
            if d_min < -100: # Should not happen if scaled
                return -d_min
            
            # log-sum-exp stability
            # soft_min = -1/beta * log(sum(exp(-beta * d)))
            # = -1/beta * log(sum(exp(-beta * (d - d_min) * beta? No.
            # Let u = -beta * d.
            # max_u = max(u) = -beta * min(d)
            # sum_exp = sum(exp(u - max_u))
            # soft_min = -1/beta * (max_u + log(sum_exp))
            
            u = -beta * all_dists
            max_u = np.max(u)
            sum_exp = np.sum(np.exp(u - max_u))
            soft_min_val = -(max_u + np.log(sum_exp)) / beta
            
            return -soft_min_val # Minimize negative
        
        # Optimization
        best_x = x0
        best_score = -objective(x0.flatten())
        
        # Run L-BFGS-B
        # Use a few phases with increasing beta
        betas = [50, 200, 1000]
        
        for beta_val in betas:
            if time.time() > deadline - 10:
                break
                
            # Redefine objective with current beta
            def obj_curr(x_flat):
                x = x_flat.reshape(N, 2)
                diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dists, np.inf)
                
                dist_x = 0.5 - np.abs(x[:, 0])
                dist_y = 0.5 - np.abs(x[:, 1])
                boundary_dists = np.minimum(dist_x, dist_y)
                
                all_dists = np.concatenate([dists.flatten(), boundary_dists])
                all_dists = all_dists[np.isfinite(all_dists)]
                
                if len(all_dists) == 0:
                    return 0.0
                
                u = -beta_val * all_dists
                max_u = np.max(u)
                sum_exp = np.sum(np.exp(u - max_u))
                soft_min_val = -(max_u + np.log(sum_exp)) / beta_val
                
                return -soft_min_val

            bounds = [(-0.5, 0.5)] * (2 * N)
            
            try:
                res = minimize(obj_curr, best_x.flatten(), method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12})
                if res.success or res.fun < -best_score:
                    best_x = res.x.reshape(N, 2)
                    best_score = -res.fun
            except Exception:
                pass
        
        return best_x
