"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _objective(params, w):
    """
    Objective function to minimize.
    params: array of shape (3*N,) -> [x0, x1, ..., xN-1, y0, y1, ..., yN-1, r0, r1, ..., rN-1]
    w: width of the rectangle
    """
    h = 2.0 - w
    
    x = params[0:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    
    # 1. Score to maximize: sum(r)
    score = np.sum(r)
    
    # 2. Penalties
    penalty = 0.0
    
    # Boundary violations
    d_left = x
    d_right = w - x
    d_bottom = y
    d_top = h - y
    
    min_dist = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    boundary_viol = np.maximum(0, r - min_dist)
    penalty += np.sum(boundary_viol ** 2)
    
    # Overlap violations
    diff_x = x[:, np.newaxis] - x[np.newaxis, :]
    diff_y = y[:, np.newaxis] - y[np.newaxis, :]
    dists = np.sqrt(diff_x**2 + diff_y**2)
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    mask = ~np.eye(N, dtype=bool)
    
    violations = np.maximum(0, r_sum - dists)
    violations = violations * mask
    penalty += np.sum(violations ** 2)
    
    # We want to minimize: Penalty - Score
    # Large penalty weight to ensure constraints are met
    return penalty * 10.0 - score

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_score = -np.inf
    best_result = None
    
    # Try a range of widths
    widths = [0.9, 0.95, 1.0, 1.05, 1.1]
    
    for w in widths:
        if time.time() - start_time > time_limit:
            break
            
        h = 2.0 - w
        
        # Multi-start for this width
        for restart in range(3):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize in a grid
            cols = int(np.sqrt(N * w / h)) + 1
            rows = (N + cols - 1) // cols
            
            x_init = np.zeros(N)
            y_init = np.zeros(N)
            r_init = np.zeros(N)
            
            for i in range(N):
                row = i // cols
                col = i % cols
                x_init[i] = (col + 0.5) * w / cols
                y_init[i] = (row + 0.5) * h / rows
                cell_w = w / cols
                cell_h = h / rows
                r_init[i] = 0.4 * min(cell_w, cell_h)
            
            # Add random perturbation to break symmetry
            if restart > 0:
                x_init += np.random.uniform(-0.05, 0.05, N)
                y_init += np.random.uniform(-0.05, 0.05, N)
                
            params_init = np.concatenate([x_init, y_init, r_init])
            
            bounds = []
            for i in range(N):
                bounds.append((0, w))
            for i in range(N):
                bounds.append((0, h))
            for i in range(N):
                bounds.append((0, 1.0))
                
            try:
                res = minimize(
                    _objective, 
                    params_init, 
                    args=(w,), 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if res.success or res.fun < best_score:
                    params_opt = res.x
                    x = params_opt[0:N]
                    y = params_opt[N:2*N]
                    r = params_opt[2*N:3*N]
                    
                    # Validate and clean up
                    valid = True
                    for i in range(N):
                        if x[i] < -1e-8 or x[i] > w + 1e-8 or y[i] < -1e-8 or y[i] > h + 1e-8:
                            valid = False
                            break
                        max_r = min(x[i], w - x[i], y[i], h - y[i])
                        if r[i] > max_r + 1e-8:
                            valid = False
                            break
                    if not valid:
                        continue
                        
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                            if dist < r[i] + r[j] - 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                        
                    current_score = np.sum(r)
                    if current_score > best_score:
                        best_score = current_score
                        result = np.column_stack([x, y, r])
                        best_result = result
                        
            except Exception as e:
                pass

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append([x, y, r])
        return np.array(cells)
        
    return best_result

# EVOLVE-BLOCK-END
