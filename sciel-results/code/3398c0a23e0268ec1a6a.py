# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _build_grid():
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return X.ravel(), Y.ravel()

def _build_subgrid(step=2):
    xs = np.arange(-3.0, 1.5, 0.01 * step)
    ys = np.arange(-1.5, 1.5, 0.01 * step)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return X.ravel(), Y.ravel()

def _densify_path(path):
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
        
        arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        in_base = arm1 | arm2
        
        inside = inside & in_base
        if not np.any(inside):
            break
    
    return np.sum(inside) * 0.0001

def _evaluate_path_sub(path, sub_x, sub_y, full_x, full_y, scale_factor):
    """Evaluate using subgrid for speed, then validate with full grid if promising."""
    area_sub = _evaluate_path(path, sub_x, sub_y)
    # Scale back: subgrid has fewer points, so area estimate is rough
    # But we can use it for comparison during optimization
    return area_sub

def _negative_area(params, grid_x, grid_y):
    n = len(params) // 3
    path = params.reshape(n, 3)
    area = _evaluate_path(path, grid_x, grid_y)
    return -area

def _negative_area_sub(params, sub_x, sub_y):
    n = len(params) // 3
    path = params.reshape(n, 3)
    area = _evaluate_path(path, sub_x, sub_y)
    return -area

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 570  # 570s safety margin
    
    grid_x, grid_y = _build_grid()
    sub_x, sub_y = _build_subgrid(step=3)
    
    best_params = None
    best_area = 0.0
    best_n_ctrl = 8
    
    # Phase 1: Quick search with subgrid
    for n_ctrl in [6, 8, 10, 12]:
        if time.time() > deadline - 60:
            break
        
        t = np.linspace(0, 1, n_ctrl)
        angles = np.linspace(5, 85, n_ctrl)
        x = -2.0 * t
        y = 0.5 * np.sin(np.pi * t)
        
        initial_params = np.column_stack([x, y, angles]).flatten()
        
        # Try a few different starting y-offsets
        for y_offset in [0.0, 0.3, -0.3, 0.5, -0.5]:
            if time.time() > deadline - 60:
                break
            
            params_try = initial_params.copy()
            params_try[1::3] += y_offset  # adjust y values
            
            try:
                result = minimize(
                    _negative_area_sub,
                    params_try,
                    args=(sub_x, sub_y),
                    method='Nelder-Mead',
                    options={'maxiter': 150, 'xatol': 0.02, 'fatol': 0.001}
                )
                
                # Validate with full grid
                area_full = _evaluate_path(result.x.reshape(n_ctrl, 3), grid_x, grid_y)
                
                if area_full > best_area:
                    best_area = area_full
                    best_params = result.x.copy()
                    best_n_ctrl = n_ctrl
            except:
                pass
    
    if best_params is None:
        # Fallback to a simple path
        n_ctrl = 8
        t = np.linspace(0, 1, n_ctrl)
        angles = np.linspace(10, 80, n_ctrl)
        x = -2.0 * t
        y = 0.5 * np.sin(np.pi * t)
        best_params = np.column_stack([x, y, angles]).flatten()
        best_n_ctrl = n_ctrl
        best_area = _evaluate_path(best_params.reshape(n_ctrl, 3), grid_x, grid_y)
    
    # Phase 2: Refine with full grid
    if time.time() < deadline - 30:
        try:
            result = minimize(
                _negative_area,
                best_params,
                args=(grid_x, grid_y),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 0.005, 'fatol': 0.0001}
            )
            area = _evaluate_path(result.x.reshape(best_n_ctrl, 3), grid_x, grid_y)
            if area > best_area:
                best_area = area
                best_params = result.x.copy()
        except:
            pass
    
    # Phase 3: Final local refinement
    if time.time() < deadline - 10:
        try:
            result = minimize(
                _negative_area,
                best_params,
                args=(grid_x, grid_y),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 0.002, 'fatol': 0.00005}
            )
            area = _evaluate_path(result.x.reshape(best_n_ctrl, 3), grid_x, grid_y)
            if area > best_area:
                best_area = area
                best_params = result.x.copy()
        except:
            pass
    
    return best_params.reshape(best_n_ctrl, 3)

# EVOLVE-BLOCK-END
