# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _build_grid():
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return X.ravel(), Y.ravel()

def _build_subgrid(step=3):
    xs = np.arange(-3.0, 1.5, 0.01 * step)
    ys = np.arange(-1.5, 1.5, 0.01 * step)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return X.ravel(), Y.ravel()

def _densify_path(path):
    # Prepend (0,0,0) and append (-2,0,90)
    full = np.vstack([[0.0, 0.0, 0.0], path, [-2.0, 0.0, 90.0]])
    result = [full[0]]
    for i in range(len(full) - 1):
        p1 = full[i]
        p2 = full[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        for k in range(1, m + 1):
            t = k / m
            interp = p1 + t * (p2 - p1)
            result.append(interp)
    return np.array(result)

def _evaluate_path(path, grid_x, grid_y):
    dense = _densify_path(path)
    n_points = len(grid_x)
    inside = np.ones(n_points, dtype=bool)
    
    for i in range(len(dense)):
        x, y, a_deg = dense[i]
        a_rad = np.deg2rad(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        dx = grid_x - x
        dy = grid_y - y
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        # Base L-shape: {0 <= u <= 1, -3 <= v <= 1} U {-3 <= u <= 1, 0 <= v <= 1}
        arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        in_base = arm1 | arm2
        
        inside = inside & in_base
        if not np.any(inside):
            break
    
    return np.sum(inside) * 0.0001

def _negative_area_sub(params, sub_x, sub_y, n_ctrl):
    path = params.reshape(n_ctrl, 3)
    area = _evaluate_path(path, sub_x, sub_y)
    return -area

def _negative_area_full(params, full_x, full_y, n_ctrl):
    path = params.reshape(n_ctrl, 3)
    area = _evaluate_path(path, full_x, full_y)
    return -area

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Build grids
    full_x, full_y = _build_grid()
    sub_x, sub_y = _build_subgrid(step=3)
    
    best_params = None
    best_area = 0.0
    best_n_ctrl = 10
    
    # Try different numbers of control points
    for n_ctrl in [8, 10, 12]:
        if time.time() > deadline - 60:
            break
        
        # Initial guess: linear path in x, y, a
        t = np.linspace(0, 1, n_ctrl)
        # Start near (0,0), end near (-2,0)
        # The sofa rotates 90 degrees.
        # A common heuristic is to move the center along a curve that avoids the corners.
        x_init = -2.0 * t
        y_init = 0.5 * np.sin(np.pi * t) # Bulge out to avoid corners
        a_init = 90.0 * t
        
        initial_params = np.column_stack([x_init, y_init, a_init]).flatten()
        
        # Optimization on Subgrid
        try:
            # Use Nelder-Mead on subgrid
            result = minimize(
                _negative_area_sub,
                initial_params,
                args=(sub_x, sub_y, n_ctrl),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 0.02, 'fatol': 0.001, 'adaptive': True}
            )
            
            # Validate on full grid
            area_full = _evaluate_path(result.x.reshape(n_ctrl, 3), full_x, full_y)
            
            if area_full > best_area:
                best_area = area_full
                best_params = result.x.copy()
                best_n_ctrl = n_ctrl
        except Exception:
            pass
    
    if best_params is None:
        # Fallback
        n_ctrl = 10
        t = np.linspace(0, 1, n_ctrl)
        x_init = -2.0 * t
        y_init = 0.5 * np.sin(np.pi * t)
        a_init = 90.0 * t
        best_params = np.column_stack([x_init, y_init, a_init]).flatten()
        best_n_ctrl = n_ctrl
        best_area = _evaluate_path(best_params.reshape(n_ctrl, 3), full_x, full_y)
    
    # Refinement on Full Grid
    if time.time() < deadline - 30:
        try:
            result = minimize(
                _negative_area_full,
                best_params,
                args=(full_x, full_y, best_n_ctrl),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 0.005, 'fatol': 0.0001, 'adaptive': True}
            )
            area = _evaluate_path(result.x.reshape(best_n_ctrl, 3), full_x, full_y)
            if area > best_area:
                best_area = area
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Final fine-tuning on Full Grid
    if time.time() < deadline - 10:
        try:
            result = minimize(
                _negative_area_full,
                best_params,
                args=(full_x, full_y, best_n_ctrl),
                method='Powell',
                options={'maxiter': 50, 'ftol': 1e-6, 'xtol': 1e-4}
            )
            area = _evaluate_path(result.x.reshape(best_n_ctrl, 3), full_x, full_y)
            if area > best_area:
                best_area = area
                best_params = result.x.copy()
        except Exception:
            pass

    return best_params.reshape(best_n_ctrl, 3)

# EVOLVE-BLOCK-END
