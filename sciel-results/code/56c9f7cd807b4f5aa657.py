"""Circle packing in perimeter-4 rectangle using soft-body penalty + L-BFGS-B."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _penalty_objective(params, w):
    """
    params: array of shape (2*N + N) = [x_0..x_{N-1}, y_0..y_{N-1}, r_0..r_{N-1}]
    w: width of rectangle, h = 2 - w
    Returns: penalty - sum(r_i)
    """
    h = 2.0 - w
    x = params[:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    
    # Objective: maximize sum(r), so minimize -sum(r)
    obj = -np.sum(r)
    
    # Wall penalties (quadratic)
    # Left wall
    pen_left = np.maximum(0, r - x)
    obj += 1000.0 * np.sum(pen_left ** 2)
    # Right wall
    pen_right = np.maximum(0, r - (w - x))
    obj += 1000.0 * np.sum(pen_right ** 2)
    # Bottom wall
    pen_bottom = np.maximum(0, r - y)
    obj += 1000.0 * np.sum(pen_bottom ** 2)
    # Top wall
    pen_top = np.maximum(0, r - (h - y))
    obj += 1000.0 * np.sum(pen_top ** 2)
    
    # Circle-circle overlap penalties (quadratic)
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            min_dist = r[i] + r[j]
            # Overlap if dist < min_dist
            dist = np.sqrt(dist_sq)
            overlap = np.maximum(0, min_dist - dist)
            obj += 1000.0 * overlap ** 2
    
    # Penalty for negative radii
    pen_neg_r = np.maximum(0, -r)
    obj += 1000.0 * np.sum(pen_neg_r ** 2)
    
    return obj

def _validate(w, h, x, y, r, tol=1e-9):
    """Check if the packing is valid (no overlaps, within bounds)."""
    if w < 0.1 or h < 0.1:
        return False
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(w - x - r < -tol):
        return False
    if np.any(y - r < -tol) or np.any(h - y - r < -tol):
        return False
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            if dist_sq < (r[i] + r[j] - tol) ** 2:
                return False
    return True

def _grid_init(w):
    """Create a grid initialization for N circles in rectangle w x (2-w)."""
    h = 2.0 - w
    aspect = w / h
    if aspect >= 1.0:
        n_cols = int(np.round(np.sqrt(N * aspect)))
        n_rows = int(np.ceil(N / n_cols))
    else:
        n_rows = int(np.round(np.sqrt(N / aspect)))
        n_cols = int(np.ceil(N / n_rows))
    
    while n_cols * n_rows < N:
        if aspect >= 1.0:
            n_cols += 1
        else:
            n_rows += 1
    
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    cell_w = w / n_cols
    cell_h = h / n_rows
    max_r = min(cell_w, cell_h) * 0.45
    
    idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if idx >= N:
                break
            x[idx] = (j + 0.5) * cell_w
            y[idx] = (i + 0.5) * cell_h
            r[idx] = max_r
            idx += 1
    return x, y, r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_result = None
    best_score = -np.inf
    
    def update_best(w, h, x, y, r):
        nonlocal best_result, best_score
        if _validate(w, h, x, y, r):
            x = np.clip(x, 0, w)
            y = np.clip(y, 0, h)
            r = np.clip(r, 0, None)
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([x, y, r])
    
    # Try multiple aspect ratios
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 30.0:
            break
        
        # w/h = ar, w + h = 2 => w = 2*ar/(1+ar)
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 - w
        
        # Grid initialization
        x, y, r = _grid_init(w)
        update_best(w, h, x, y, r)
        
        # Build parameter vector
        params = np.concatenate([x, y, r])
        
        # Define bounds
        bounds = []
        for i in range(N):
            bounds.append((0, w))  # x
        for i in range(N):
            bounds.append((0, h))  # y
        for i in range(N):
            bounds.append((0, min(w, h)/2))  # r
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                _penalty_objective,
                params,
                args=(w,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < _penalty_objective(params, w):
                opt_params = result.x
                opt_x = opt_params[:N]
                opt_y = opt_params[N:2*N]
                opt_r = opt_params[2*N:3*N]
                update_best(w, h, opt_x, opt_y, opt_r)
        except Exception:
            pass
        
        # Try a second optimization with perturbed initial conditions
        if time.time() > deadline - 30.0:
            break
        
        x2, y2, r2 = _grid_init(w)
        # Add small random perturbation
        x2 += np.random.normal(0, 0.02, N)
        y2 += np.random.normal(0, 0.02, N)
        x2 = np.clip(x2, 0, w)
        y2 = np.clip(y2, 0, h)
        
        params2 = np.concatenate([x2, y2, r2])
        
        try:
            result2 = minimize(
                _penalty_objective,
                params2,
                args=(w,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result2.success or result2.fun < _penalty_objective(params2, w):
                opt_params2 = result2.x
                opt_x2 = opt_params2[:N]
                opt_y2 = opt_params2[N:2*N]
                opt_r2 = opt_params2[2*N:3*N]
                update_best(w, h, opt_x2, opt_y2, opt_r2)
        except Exception:
            pass
    
    # Final fallback
    if best_result is None:
        x, y, r = _grid_init(1.0)
        best_result = np.column_stack([x, y, r])
    
    return best_result
