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
        # Vectorized calculation of pairwise distances
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
        lambda_pen = 100.0
        return -np.sum(r) + lambda_pen * total_pen

    def grid_init():
        # 5x5 grid + 1 center
        coords = []
        side = 5
        for i in range(side):
            for j in range(side):
                if len(coords) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.1
                    coords.append([x, y, r])
        while len(coords) < N:
            coords.append([0.5, 0.5, 0.1])
        arr = np.array(coords[:N])
        # Add jitter
        arr[:, 0] += np.random.uniform(-0.02, 0.02, N)
        arr[:, 1] += np.random.uniform(-0.02, 0.02, N)
        return np.concatenate([arr[:, 0], arr[:, 1], arr[:, 2]])

    # Define bounds for Differential Evolution
    # x, y in [0, 1], r in [0, 0.25]
    bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)] * N + [(0.0, 0.25)] * N
    
    best_val = -np.inf
    best_params = None
    
    # Run Differential Evolution
    # We use a small population and moderate iterations to stay within time limits
    # and rely on the polishing step for final accuracy.
    try:
        res_de = differential_evolution(
            objective, 
            bounds, 
            maxiter=300, 
            popsize=15, 
            tol=1e-4, 
            seed=42,
            polish=False
        )
        if res_de.fun < best_val:
            best_val = res_de.fun
            best_params = res_de.x
    except Exception as e:
        pass

    # Polish with Nelder-Mead to refine the solution found by DE
    if best_params is not None:
        try:
            res_nm = minimize(
                objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_params = res_nm.x
        except Exception as e:
            pass

    if best_params is None:
        # Fallback to seed-like grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

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
            v = max(0.0, -min_x, max_x - 1.0, -min_y, max_y - 1.0)
            violation = max(violation, v)
            
        # Check overlaps
        overlaps = np.maximum(0, r_sum - d)
        max_overlap = np.max(overlaps)
        if max_overlap > 1e-8:
            violation = max(violation, max_overlap)
            
        if violation > 1e-8:
            # Shrink radii
            # We want to reduce r such that violations are resolved.
            # Simple heuristic: reduce all r by a factor proportional to violation
            shrink_factor = 1.0 - (violation / (2.0 * np.max(r) + 1e-6))
            shrink_factor = max(0.9, min(1.0, shrink_factor))
            r = r * shrink_factor
        else:
            break
            
    # Final clip to ensure strictly inside
    r = np.maximum(r, 1e-6)
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    
    # One last check to ensure no overlaps after clipping
    for _ in range(5):
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
