import numpy as np
import time
from scipy.optimize import minimize

# Grid definition
X, Y = np.meshgrid(np.arange(-3, 1.5, 0.01), np.arange(-1.5, 1.5, 0.01))
GRID_POINTS = np.column_stack((X.ravel(), Y.ravel()))
N_POINTS = len(GRID_POINTS)

# Precompute start and end
START = np.array([0.0, 0.0, 0.0])
END = np.array([-2.0, 0.0, 90.0])

def _get_corridor_mask(x, y, a_deg):
    """
    Returns a boolean mask of shape (N_POINTS,) indicating which grid points
    are inside the corridor placement (x, y, a_deg).
    """
    a_rad = np.deg2rad(a_deg)
    cos_a = np.cos(-a_rad)
    sin_a = np.sin(-a_rad)
    
    dx = GRID_POINTS[:, 0] - x
    dy = GRID_POINTS[:, 1] - y
    
    u = cos_a * dx - sin_a * dy
    v = sin_a * dx + cos_a * dy
    
    # Base corridor conditions
    mask1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    mask2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    
    return mask1 | mask2

def _densify_path(path):
    """
    Densify path per evaluator rules: add linear interpolations between consecutive rows.
    """
    # path is (n, 3)
    # Prepend start, append end
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
            # Interpolate m steps
            # t goes from 1/m to 1
            for k in range(1, m + 1):
                t = k / m
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                a = a1 + t * (a2 - a1)
                dense.append([x, y, a])
        else:
            # m == 1, just add p2
            dense.append(p2)
            
    return np.array(dense)

def _compute_score(path):
    """
    Compute the raw score for a given path (intermediate points).
    """
    dense = _densify_path(path)
    
    # Initialize mask with all True
    mask = np.ones(N_POINTS, dtype=bool)
    
    # Apply each placement
    for i in range(len(dense)):
        x, y, a = dense[i]
        step_mask = _get_corridor_mask(x, y, a)
        mask &= step_mask
        
        # Early exit if no points remain
        if not mask.any():
            break
            
    return np.sum(mask) * 0.0001

def _generate_arc_path(cy, n_points=15):
    """
    Generate a path following a circular arc from (0,0) to (-2,0)
    with center at (-1, cy).
    """
    cx = -1.0
    R = np.sqrt(1.0 + cy**2)
    
    # Angles of start and end points relative to center
    # Start: (0,0) -> vector (1, -cy)
    theta_start = np.arctan2(-cy, 1.0)
    # End: (-2,0) -> vector (-1, -cy)
    theta_end = np.arctan2(-cy, -1.0)
    
    # Determine the arc direction
    d_theta = theta_end - theta_start
    # Normalize to (-pi, pi]
    d_theta = (d_theta + np.pi) % (2 * np.pi) - np.pi
    
    # Parametrize by t in [0, 1]
    thetas = theta_start + np.linspace(0, 1, n_points + 1) * d_theta
    
    # Generate points
    xs = cx + R * np.cos(thetas)
    ys = cy + R * np.sin(thetas)
    
    # Filter out endpoints (we only return intermediate points)
    # The evaluator adds start and end.
    # So we return points for t in (0, 1)
    idx = np.arange(1, n_points)
    xs = xs[idx]
    ys = ys[idx]
    
    # Angles a
    # Linear interpolation of angle from 0 to 90
    a_vals = np.linspace(0, 90, n_points + 1)[idx]
    
    path = np.column_stack([xs, ys, a_vals])
    return path

def objective(params):
    """
    Minimize negative score.
    params: [cy]
    """
    cy = params[0]
    # Optional: clamp cy to reasonable range
    cy = np.clip(cy, -4.0, 4.0)
    
    path = _generate_arc_path(cy, n_points=15)
    score = _compute_score(path)
    return -score

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 550  # Safety margin
    
    # Initial guess
    # Try cy = 0 (semicircle)
    best_params = np.array([0.0])
    best_score = -objective(best_params)
    
    # Grid search for cy - expanded range and more points
    cy_range = np.linspace(-3.0, 3.0, 25)
    for cy in cy_range:
        if time.time() > deadline:
            break
        p = np.array([cy])
        score = -objective(p)
        if score > best_score:
            best_score = score
            best_params = p
    
    # Local optimization - first pass
    if time.time() < deadline - 60:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5, 'disp': False})
            if res.fun < -best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass
        
        # Second refinement with tighter tol
        if time.time() < deadline - 30:
            try:
                res2 = minimize(objective, best_params, method='Nelder-Mead', 
                                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False})
                if res2.fun < -best_score:
                    best_score = -res2.fun
                    best_params = res2.x
            except Exception:
                pass

    # Generate final path
    cy_opt = best_params[0]
    path = _generate_arc_path(cy_opt, n_points=15)
    
    # Ensure finite
    if not np.all(np.isfinite(path)):
        path = _generate_arc_path(0.0, n_points=15)
        
    return path

# EVOLVE-BLOCK-END
