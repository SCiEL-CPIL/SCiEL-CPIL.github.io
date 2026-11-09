import numpy as np
import time
from scipy.optimize import minimize

# Grid definition
X, Y = np.meshgrid(np.arange(-3, 1.5, 0.01), np.arange(-1.5, 1.5, 0.01))
GRID_POINTS = np.column_stack((X.ravel(), Y.ravel()))
N_POINTS = len(GRID_POINTS)

START = np.array([0.0, 0.0, 0.0])
END = np.array([-2.0, 0.0, 90.0])

def _densify_path(path):
    """
    Densify path per evaluator rules: add linear interpolations between consecutive rows.
    Returns a numpy array of shape (M, 3).
    """
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
    """
    Compute the raw score using vectorized operations.
    """
    dense = _densify_path(path)
    M = len(dense)
    
    x = dense[:, 0]
    y = dense[:, 1]
    a = dense[:, 2]
    
    # Broadcasting: 
    # GRID_POINTS is (N, 2). 
    # We want u, v of shape (M, N).
    
    dx = GRID_POINTS[:, 0][np.newaxis, :] - x[:, np.newaxis] # (1, N) - (M, 1) -> (M, N)
    dy = GRID_POINTS[:, 1][np.newaxis, :] - y[:, np.newaxis] # (1, N) - (M, 1) -> (M, N)
    
    a_rad = np.deg2rad(a)
    cos_a = np.cos(-a_rad)[:, np.newaxis] # (M, 1)
    sin_a = np.sin(-a_rad)[:, np.newaxis] # (M, 1)
    
    u = cos_a * dx - sin_a * dy
    v = sin_a * dx + cos_a * dy
    
    # Mask conditions
    # mask1: 0 <= u <= 1, -3 <= v <= 1
    # mask2: -3 <= u <= 1, 0 <= v <= 1
    
    m1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    m2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    
    m_union = m1 | m2
    
    # Intersection over all placements (axis 0)
    mask_final = np.all(m_union, axis=0)
    
    return np.sum(mask_final) * 0.0001

def _generate_arc_path(c_y, n_points=50):
    """
    Generate a path following a circular arc from (0,0) to (-2,0)
    with center at (-1, c_y).
    """
    cx = -1.0
    R = np.sqrt(1.0 + c_y**2)
    
    # Angles of start and end points relative to center
    theta_start = np.arctan2(-c_y, 1.0)
    theta_end = np.arctan2(-c_y, -1.0)
    
    thetas = theta_start + np.linspace(0, 1, n_points + 1) * (theta_end - theta_start)
    
    xs = cx + R * np.cos(thetas)
    ys = c_y + R * np.sin(thetas)
    
    # Filter out endpoints
    idx = np.arange(1, n_points)
    xs = xs[idx]
    ys = ys[idx]
    
    # Angles a: linear 0 to 90
    a_vals = np.linspace(0, 90, n_points + 1)[idx]
    
    path = np.column_stack([xs, ys, a_vals])
    return path

def _generate_bezier_path(x1, y1, x2, y2, n_points=50):
    """
    Generate a path following a cubic Bezier curve from (0,0) to (-2,0)
    with control points (x1, y1) and (x2, y2).
    """
    P0 = np.array([0.0, 0.0])
    P1 = np.array([x1, y1])
    P2 = np.array([x2, y2])
    P3 = np.array([-2.0, 0.0])
    
    ts = np.linspace(0, 1, n_points + 1)
    ts = ts[1:-1] # Exclude endpoints
    
    # Bezier formula: (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
    t1 = 1 - ts
    c0 = t1**3
    c1 = 3 * t1**2 * ts
    c2 = 3 * t1 * ts**2
    c3 = ts**3
    
    xs = c0 * P0[0] + c1 * P1[0] + c2 * P2[0] + c3 * P3[0]
    ys = c0 * P0[1] + c1 * P1[1] + c2 * P2[1] + c3 * P3[1]
    
    a_vals = np.linspace(0, 90, n_points + 1)[1:-1]
    
    path = np.column_stack([xs, ys, a_vals])
    return path

def _generate_poly_path(coeffs, n_points=50):
    """
    Generate a path using a polynomial of degree 5 for x and y.
    coeffs: [a1, a2, a3, a4, a5, b1, b2, b3, b4, b5]
    x(t) = a1*t + ... + a5*t^5
    y(t) = b1*t + ... + b5*t^5
    Constraints: x(1) = -2, y(1) = 0.
    We enforce these by adjusting the last coefficients.
    """
    a = np.array(coeffs[:5])
    b = np.array(coeffs[5:10])
    
    # Enforce x(1) = -2
    sum_a = np.sum(a)
    a[4] += (-2.0 - sum_a)
    
    # Enforce y(1) = 0
    sum_b = np.sum(b)
    b[4] += (0.0 - sum_b)
    
    ts = np.linspace(0, 1, n_points + 1)
    ts = ts[1:-1]
    
    # Compute x and y
    # Using Horner's rule or just powers
    t_powers = np.stack([ts**i for i in range(1, 6)], axis=1) # (N, 5)
    
    xs = t_powers @ a
    ys = t_powers @ b
    
    a_vals = np.linspace(0, 90, n_points + 1)[1:-1]
    
    path = np.column_stack([xs, ys, a_vals])
    return path

def objective_arc(params):
    c_y = params[0]
    n_points = int(params[1])
    c_y = np.clip(c_y, -5.0, 5.0)
    n_points = int(np.clip(n_points, 10, 100))
    
    path = _generate_arc_path(c_y, n_points)
    score = _compute_score_fast(path)
    return -score

def objective_bezier(params):
    x1, y1, x2, y2 = params
    n_points = 50
    
    path = _generate_bezier_path(x1, y1, x2, y2, n_points)
    score = _compute_score_fast(path)
    return -score

def objective_poly(params):
    coeffs = params
    n_points = 50
    
    path = _generate_poly_path(coeffs, n_points)
    score = _compute_score_fast(path)
    return -score

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580 # Safety margin
    
    best_score = -np.inf
    best_path = None
    
    def update_best(path, score):
        nonlocal best_score, best_path
        if score > best_score:
            best_score = score
            best_path = path

    # --- Phase 1: Arc Search (Fast baseline) ---
    c_y_range = np.linspace(-2.0, 2.0, 5)
    n_points_range = [30, 50]
    
    for c_y in c_y_range:
        for n_pts in n_points_range:
            if time.time() > deadline - 100:
                break
            try:
                path = _generate_arc_path(c_y, n_pts)
                score = _compute_score_fast(path)
                update_best(path, score)
            except Exception:
                pass

    # --- Phase 2: Bezier Search (Multi-start Nelder-Mead) ---
    if time.time() < deadline - 200:
        # Define a grid of initial guesses for Bezier control points
        # P1 and P2 should generally be between x=0 and x=-2, but can go outside.
        # y can be positive or negative.
        x1_vals = [-0.5, 0.0, 0.5, 1.0, 1.5]
        y1_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
        x2_vals = [-1.5, -1.0, -0.5, 0.0, 0.5]
        y2_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        # We can't do full grid (5*5*5*5 = 625) with NM (200 iters each) in time.
        # We will pick a subset of starts.
        starts = []
        # Symmetric and asymmetric starts
        for x1 in [0.0, 0.5, 1.0]:
            for y1 in [-1.0, 0.0, 1.0]:
                for x2 in [-1.0, -1.5, -2.0]:
                    for y2 in [-1.0, 0.0, 1.0]:
                        starts.append([x1, y1, x2, y2])
        
        # Shuffle or just take first N
        import random
        random.shuffle(starts)
        
        max_starts = 20 # Limit starts to keep time under control
        for s in starts[:max_starts]:
            if time.time() > deadline - 100:
                break
            try:
                res = minimize(objective_bezier, s, method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False})
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_path = _generate_bezier_path(res.x[0], res.x[1], res.x[2], res.x[3], 50)
            except Exception:
                pass

    # --- Phase 3: Poly Search (Multi-start Nelder-Mead) ---
    if time.time() < deadline - 50:
        # Initialize coeffs to 0
        # x(t) approx linear: a1 = -2, others 0.
        # y(t) approx 0.
        init_poly = np.array([-2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        # Perturbations
        poly_starts = [init_poly]
        for i in range(5):
            p = init_poly.copy()
            p[i] += 0.5 * np.random.randn()
            p[i+5] += 0.5 * np.random.randn()
            poly_starts.append(p)
        
        for s in poly_starts:
            if time.time() > deadline - 20:
                break
            try:
                res = minimize(objective_poly, s, method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False})
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_path = _generate_poly_path(res.x, 50)
            except Exception:
                pass

    if best_path is None or not np.all(np.isfinite(best_path)):
        best_path = _generate_arc_path(0.0, 50)
        
    return best_path
