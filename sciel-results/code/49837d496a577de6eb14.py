import numpy as np
import time
from scipy.optimize import minimize

# Grid definition
X, Y = np.meshgrid(np.arange(-3, 1.5, 0.01), np.arange(-1.5, 1.5, 0.01))
GRID_POINTS = np.column_stack((X.ravel(), Y.ravel()))
N_POINTS = len(GRID_POINTS)

START = np.array([0.0, 0.0, 0.0])
END = np.array([-2.0, 0.0, 90.0])

def _get_corridor_mask(x, y, a_deg):
    a_rad = np.deg2rad(a_deg)
    cos_a = np.cos(-a_rad)
    sin_a = np.sin(-a_rad)
    
    dx = GRID_POINTS[:, 0] - x
    dy = GRID_POINTS[:, 1] - y
    
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

def _compute_score(path):
    dense = _densify_path(path)
    mask = np.ones(N_POINTS, dtype=bool)
    
    for i in range(len(dense)):
        x, y, a = dense[i]
        step_mask = _get_corridor_mask(x, y, a)
        mask &= step_mask
        if not mask.any():
            break
            
    return np.sum(mask) * 0.0001

def _generate_path_from_params(params, n_steps=20):
    # params: [A, B, C, D, E, F]
    A, B, C, D, E, F = params
    
    # Angles for intermediate points
    # We will generate points at angles 4.5, 9, ..., 85.5
    # Start is 0, End is 90.
    angles = np.linspace(4.5, 85.5, n_steps)
    
    xs = np.zeros(len(angles))
    ys = np.zeros(len(angles))
    
    for i, a in enumerate(angles):
        t = a / 90.0
        # x(a) = -2*t + A*sin(pi*t) + C*sin(2*pi*t) + E*sin(3*pi*t)
        # y(a) = B*sin(pi*t) + D*sin(2*pi*t) + F*sin(3*pi*t)
        xs[i] = -2.0 * t + A * np.sin(np.pi * t) + C * np.sin(2 * np.pi * t) + E * np.sin(3 * np.pi * t)
        ys[i] = B * np.sin(np.pi * t) + D * np.sin(2 * np.pi * t) + F * np.sin(3 * np.pi * t)
        
    path = np.column_stack([xs, ys, angles])
    return path

def objective(params):
    path = _generate_path_from_params(params)
    score = _compute_score(path)
    return -score

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 550
    
    best_params = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    best_score = -objective(best_params)
    
    # Grid search for initial guess
    # Try a range of A and B
    for A in np.linspace(-1.0, 1.0, 5):
        for B in np.linspace(-1.0, 1.0, 5):
            if time.time() > deadline:
                break
            p = np.array([A, B, 0.0, 0.0, 0.0, 0.0])
            score = -objective(p)
            if score > best_score:
                best_score = score
                best_params = p
    
    # Local optimization
    if time.time() < deadline - 50:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5, 'disp': False})
            if res.fun < -best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass
        
        if time.time() < deadline - 20:
            try:
                res2 = minimize(objective, best_params, method='Nelder-Mead', 
                                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False})
                if res2.fun < -best_score:
                    best_score = -res2.fun
                    best_params = res2.x
            except Exception:
                pass

    path = _generate_path_from_params(best_params)
    if not np.all(np.isfinite(path)):
        path = _generate_path_from_params(np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]))
        
    return path
