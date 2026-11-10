"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def _objective_normalized(vars):
    """
    Objective function for optimization.
    Returns negative log of the normalized product of distances.
    """
    # Reconstruct points
    p = np.zeros((8, 2))
    p[1:, 0] = vars[0::2]
    p[1:, 1] = vars[1::2]
    
    # Pairwise distances
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    i, j = np.triu_indices(8, k=1)
    pair_dists = dists[i, j]
    
    D = np.max(pair_dists)
    if D < 1e-10:
        return 1e10
        
    # Check for zero distances
    if np.any(pair_dists == 0):
        return 1e10
        
    # Log of normalized distances
    # We want to maximize sum(log(d_ij / D))
    # So we minimize -sum(log(d_ij / D))
    log_score = np.sum(np.log(pair_dists / D))
    return -log_score

def construct():
    start_time = time.time()
    deadline = 140.0
    
    # Bounds for 14 variables (coords of p1..p7)
    # Scale is invariant, so we just need a box large enough to contain
    # the relative geometry.
    bounds = [(-10, 10)] * 14
    
    best_result = None
    best_score = np.inf
    best_points = None
    
    # Run Differential Evolution with a few restarts
    seeds = [42, 123, 456, 789]
    
    for seed in seeds:
        if time.time() > deadline - 10:
            break
            
        try:
            res = differential_evolution(
                _objective_normalized, 
                bounds, 
                maxiter=400, 
                popsize=30, 
                tol=1e-10, 
                seed=seed,
                init='sobol',
                polish=True,
                update_interval=50
            )
            
            # res.fun is the minimized value (-log_score)
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
                
                # Reconstruct points
                pts = np.zeros((8, 2))
                pts[1:, 0] = best_result[0::2]
                pts[1:, 1] = best_result[1::2]
                
                # Normalize to diameter 2
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                i, j = np.triu_indices(8, k=1)
                D = np.max(dists[i, j])
                if D > 0:
                    pts = pts * (2.0 / D)
                    best_points = pts
                    
        except Exception:
            continue
            
    if best_points is None:
        # Fallback to regular octagon
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return [[float(x) for x in pt] for pt in best_points]

# EVOLVE-BLOCK-END
