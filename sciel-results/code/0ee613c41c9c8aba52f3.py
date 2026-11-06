import numpy as np
import time
from scipy.optimize import minimize
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    n = 8
    num_pairs = n * (n - 1) // 2
    
    # Precompute indices for pairs
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    def get_dist_matrix(pts):
        """Calculate pairwise distances."""
        # pts is shape (n, 2)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=-1)
        dists = np.sqrt(dists_sq)
        return dists

    def objective(x):
        """
        Objective function to minimize.
        x: flattened array of 16 coordinates.
        Returns negative log-likelihood of the product of distances scaled by diameter.
        """
        if time.time() > deadline:
            return 1e100  # Penalize heavy computation if time is running out

        pts = x.reshape((n, 2))
        
        # Compute distances
        # Using the precomputed indices is faster
        diffs = pts[i_idx] - pts[j_idx]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        
        # Filter out zero distances (shouldn't happen often, but good for stability)
        dists = np.maximum(dists, 1e-12)
        
        diam = np.max(dists)
        
        if diam < 1e-12:
            return 1e100

        # Sum of log distances
        sum_log_dists = np.sum(np.log(dists))
        
        # Term: - C * log(diam)
        # We want to maximize sum_log_dists - C * log(diam)
        # So minimize: - (sum_log_dists - C * log(diam))
        val = -(sum_log_dists - num_pairs * np.log(diam))
        
        return val

    best_result = None
    best_obj = np.inf
    best_points = None

    # Initial Guess: Regular Octagon
    # Radius 1, centered at 0
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    init_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    init_x = init_pts.flatten()
    
    # Run local optimization from initial guess
    res = minimize(objective, init_x, method='Powell', options={'maxiter': 200, 'ftol': 1e-12, 'xtol': 1e-12})
    if res.fun < best_obj:
        best_obj = res.fun
        best_points = res.x
        best_result = res

    # Multi-restart with random points
    # We will run several restarts if time permits
    max_restarts = 50
    for i in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Random points in a box
        # The scale doesn't matter, but we need to avoid degenerate configurations
        # Points in [-2, 2]
        x0 = np.random.uniform(-2, 2, size=n*2)
        
        # Try to ensure some spread
        # Simple random is usually fine for Powell
        
        try:
            res = minimize(objective, x0, method='Powell', options={'maxiter': 50, 'ftol': 1e-10, 'xtol': 1e-10})
            if res.fun < best_obj:
                best_obj = res.fun
                best_points = res.x
                best_result = res
        except Exception:
            continue

    # Refine the best point found with a more strict tolerance
    if best_points is not None:
        if time.time() < deadline - 2.0:
            try:
                res_refined = minimize(objective, best_points, method='Powell', options={'maxiter': 100, 'ftol': 1e-14, 'xtol': 1e-14})
                if res_refined.fun < best_obj:
                    best_obj = res_refined.fun
                    best_points = res_refined.x
                    best_result = res_refined
            except Exception:
                pass

    # Final check and formatting
    if best_points is None:
        # Fallback to regular octagon if optimization failed
        final_pts = init_pts
    else:
        final_pts = best_points.reshape((n, 2))
        
    # Convert to list of lists of floats
    result = [[float(x), float(y)] for x, y in final_pts]
    
    return result

# EVOLVE-BLOCK-END
