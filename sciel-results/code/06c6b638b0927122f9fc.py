import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 26

def construct():
    start_time = time.time()
    timeout = 115  # Leave 5s margin for 120s limit
    
    # Objective function to minimize: -sum(r) + penalty
    def objective(params):
        # Unpack params: x_0...x_{N-1}, y_0...y_{N-1}, r_0...r_{N-1}
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Boundary penalties: circles must be inside [0,1]x[0,1]
        # x in [r, 1-r], y in [r, 1-r]
        bound_pen = 0.0
        # Left/Right
        bound_pen += np.sum(np.maximum(0, r - x)**2)
        bound_pen += np.sum(np.maximum(0, x - (1 - r))**2)
        # Bottom/Top
        bound_pen += np.sum(np.maximum(0, r - y)**2)
        bound_pen += np.sum(np.maximum(0, y - (1 - r))**2)
        
        # Overlap penalties: d_ij >= r_i + r_j
        x_diff = np.subtract.outer(x, x)
        y_diff = np.subtract.outer(y, y)
        d_sq = x_diff**2 + y_diff**2
        # Set diagonal to inf to ignore self-overlap
        np.fill_diagonal(d_sq, np.inf)
        d = np.sqrt(d_sq)
        
        r_sum = np.add.outer(r, r)
        
        # Overlap amount: max(0, r_i + r_j - d_ij)
        overlap = np.maximum(0, r_sum - d)
        # Square the overlap to make penalty smooth-ish and emphasize large overlaps
        overlap_pen = np.sum(overlap**2)
        
        total_pen = bound_pen + overlap_pen
        
        # Maximize sum of radii -> Minimize -sum(r)
        # Weight the penalty heavily
        lambda_pen = 1000.0
        return -np.sum(r) + lambda_pen * total_pen

    # Bounds for Differential Evolution
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
    for i in range(N):
        bounds.append((0.0, 1.0)) # y
    for i in range(N):
        bounds.append((0.0, 0.5)) # r

    # Stage 1: Differential Evolution
    # DE is stochastic and global. We limit maxiter to keep it fast.
    # tol is the convergence tolerance.
    # polish=False because we will do a local polish later.
    
    best_params = None
    if time.time() - start_time < timeout - 10:
        try:
            de_res = differential_evolution(
                objective, 
                bounds, 
                maxiter=100, 
                tol=1e-6, 
                popsize=15, 
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            best_params = de_res.x
        except Exception as e:
            best_params = None
    else:
        best_params = None

    # If DE failed or didn't run, use a simple grid init as fallback
    if best_params is None:
        x = np.linspace(0.1, 0.9, N)
        y = np.linspace(0.1, 0.9, N)
        r = np.full(N, 0.1)
        best_params = np.concatenate([x, y, r])

    # Stage 2: Local Refinement with L-BFGS-B
    # L-BFGS-B uses gradients (approximated) and is much faster for local optimization.
    # We use the result from DE as the starting point.
    
    if time.time() - start_time < timeout - 5:
        try:
            local_res = minimize(
                objective, 
                best_params, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if local_res.fun < objective(best_params):
                best_params = local_res.x
        except Exception as e:
            pass

    x = best_params[:N]
    y = best_params[N:2*N]
    r = best_params[2*N:3*N]
    
    # Post-processing: Ensure validity
    # If any overlap or boundary violation remains, shrink radii slightly
    max_iter = 20
    for _ in range(max_iter):
        # Check boundary
        min_x = np.min(x - r)
        max_x = np.max(x + r)
        min_y = np.min(y - r)
        max_y = np.max(y + r)
        
        # Check overlap
        x_diff = np.subtract.outer(x, x)
        y_diff = np.subtract.outer(y, y)
        d_sq = x_diff**2 + y_diff**2
        np.fill_diagonal(d_sq, np.inf)
        d = np.sqrt(d_sq)
        r_sum = np.add.outer(r, r)
        
        # Find max violation
        violation = 0.0
        if min_x < 0 or max_x > 1 or min_y < 0 or max_y > 1:
            violation = 0.01
            
        overlaps = np.maximum(0, r_sum - d)
        max_overlap = np.max(overlaps)
        if max_overlap > 1e-6:
            violation = max_overlap
            
        if violation > 1e-6:
            # Shrink radii by 1%
            r = r * 0.99
        else:
            break
            
    # Final clip to ensure strictly inside
    r = np.maximum(r, 1e-6)
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    
    # One last check to ensure no overlaps after clipping
    for _ in range(10):
        x_diff = np.subtract.outer(x, x)
        y_diff = np.subtract.outer(y, y)
        d_sq = x_diff**2 + y_diff**2
        np.fill_diagonal(d_sq, np.inf)
        d = np.sqrt(d_sq)
        r_sum = np.add.outer(r, r)
        overlaps = np.maximum(0, r_sum - d)
        if np.max(overlaps) < 1e-6:
            break
        r = r * 0.99
        
    return np.column_stack((x, y, r))
