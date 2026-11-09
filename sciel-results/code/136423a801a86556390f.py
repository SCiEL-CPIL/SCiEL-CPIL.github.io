import numpy as np
import time
from scipy.optimize import minimize

# Grid definition
X, Y = np.meshgrid(np.arange(-3, 1.5, 0.01), np.arange(-1.5, 1.5, 0.01))
GRID_POINTS = np.column_stack((X.ravel(), Y.ravel()))
N_POINTS = len(GRID_POINTS)
GX = GRID_POINTS[:, 0]
GY = GRID_POINTS[:, 1]

START = np.array([0.0, 0.0, 0.0])
END = np.array([-2.0, 0.0, 90.0])

def _get_corridor_mask_fast(x, y, a_deg):
    """Vectorized corridor mask computation."""
    a_rad = np.deg2rad(a_deg)
    cos_a = np.cos(-a_rad)
    sin_a = np.sin(-a_rad)
    
    dx = GX - x
    dy = GY - y
    
    u = cos_a * dx - sin_a * dy
    v = sin_a * dx + cos_a * dy
    
    mask1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    mask2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    
    return mask1 | mask2

def _densify_path(path):
    p = np.vstack([START, path, END])
    dense = [p[0]]
    for i in range(len(p) - 1):
        p1 = p[i]
        p2 = p[i+1]
        a1 = p1[2]
        a2 = p2[2]
        delta_a = abs(a2 - a1)
        m = max(1, int(np.ceil(delta_a - 1e-9)))
        
        if m > 1:
            for k in range(1, m + 1):
                t = k / m
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                a = a1 + t * (a2 - a1)
                dense.append([x, y, a])
        else:
            dense.append(p2)
            
    return np.array(dense)

def _compute_score_fast(path):
    """Compute score using vectorized mask operations."""
    dense = _densify_path(path)
    mask = np.ones(N_POINTS, dtype=bool)
    
    for i in range(len(dense)):
        x, y, a = dense[i]
        step_mask = _get_corridor_mask_fast(x, y, a)
        mask &= step_mask
        if not mask.any():
            break
            
    return np.sum(mask) * 0.0001

def _generate_path_from_params(params, n_steps=30):
    """
    Generate path from 8 parameters: cubic Bezier control points for x and y.
    params = [x0, x1, x2, x3, y0, y1, y2, y3]
    where x(t) = (1-t)^3*x0 + 3*(1-t)^2*t*x1 + 3*(1-t)*t^2*x2 + t^3*x3
    and y(t) similarly.
    t = a/90, a in [0, 90].
    """
    x0, x1, x2, x3, y0, y1, y2, y3 = params
    
    angles = np.linspace(4.5, 85.5, n_steps)
    t = angles / 90.0
    
    # Cubic Bezier basis
    b0 = (1 - t)**3
    b1 = 3 * (1 - t)**2 * t
    b2 = 3 * (1 - t) * t**2
    b3 = t**3
    
    xs = b0 * x0 + b1 * x1 + b2 * x2 + b3 * x3
    ys = b0 * y0 + b1 * y1 + b2 * y2 + b3 * y3
    
    path = np.column_stack([xs, ys, angles])
    return path

def objective(params):
    path = _generate_path_from_params(params)
    score = _compute_score_fast(path)
    return -score

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 550
    
    # Initial guess: Bezier from (0,0) to (-2,0)
    # x: 0 -> -2, y: 0 -> 0
    # Control points for smooth curve
    best_params = np.array([0.0, -0.5, -1.5, -2.0, 0.0, 0.3, 0.3, 0.0])
    best_score = -objective(best_params)
    
    # Coarse grid search over a subset of parameters
    # Focus on the control points that affect the shape most
    search_ranges = {
        'x1': np.linspace(-1.0, 0.0, 4),
        'x2': np.linspace(-2.0, -1.0, 4),
        'y1': np.linspace(-0.5, 0.5, 4),
        'y2': np.linspace(-0.5, 0.5, 4),
    }
    
    # Coarse search
    for x1 in search_ranges['x1']:
        for x2 in search_ranges['x2']:
            for y1 in search_ranges['y1']:
                for y2 in search_ranges['y2']:
                    if time.time() > deadline - 60:
                        break
                    p = np.array([0.0, x1, x2, -2.0, 0.0, y1, y2, 0.0])
                    score = -objective(p)
                    if score > best_score:
                        best_score = score
                        best_params = p
    
    # Local optimization with Nelder-Mead
    if time.time() < deadline - 40:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False})
            if res.fun < -best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass
        
        # High-precision polish
        if time.time() < deadline - 15:
            try:
                res2 = minimize(objective, best_params, method='Nelder-Mead', 
                                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'disp': False})
                if res2.fun < -best_score:
                    best_score = -res2.fun
                    best_params = res2.x
            except Exception:
                pass

    path = _generate_path_from_params(best_params)
    if not np.all(np.isfinite(path)):
        path = _generate_path_from_params(np.array([0.0, -0.5, -1.5, -2.0, 0.0, 0.3, 0.3, 0.0]))
        
    return path
