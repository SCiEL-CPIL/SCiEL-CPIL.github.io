import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _validate(circles, w, h, tol=1e-5):
    """Check if circles are valid: inside rectangle and non-overlapping."""
    n = len(circles)
    if n != N:
        return False
    for i in range(n):
        cx, cy, r = circles[i]
        if r <= 0:
            return False
        if cx - r < -tol or cx + r > w + tol or cy - r < -tol or cy + r > h + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True

def _total_radius(circles):
    return float(np.sum(circles[:, 2]))

def _objective(params):
    """Minimize negative sum of radii."""
    r = params[2::3]
    return -np.sum(r)

def _jac_objective(params):
    """Gradient of objective function."""
    n = len(params) // 3
    jac = np.zeros(n * 3)
    # d(-sum(r))/dr_i = -1
    jac[2::3] = -1.0
    return jac

def _make_constraints(w, h):
    """
    Create constraint functions and their Jacobians for SLSQP.
    Variables: [x0, y0, r0, x1, y1, r1, ..., xN-1, yN-1, rN-1]
    """
    n = N
    
    def boundary_constraints(params):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        # Constraints:
        # x_i - r_i >= 0
        # y_i - r_i >= 0
        # w - x_i - r_i >= 0
        # h - y_i - r_i >= 0
        c1 = x - r
        c2 = y - r
        c3 = w - x - r
        c4 = h - y - r
        return np.concatenate([c1, c2, c3, c4])
    
    def boundary_jac(params):
        # Jacobian is constant for boundary constraints
        # Shape: (4*N, 3*N)
        J = np.zeros((4 * n, 3 * n))
        # c1 = x - r
        # d(c1_i)/dx_i = 1, d(c1_i)/dr_i = -1
        for i in range(n):
            J[i, 3*i] = 1.0
            J[i, 3*i+2] = -1.0
        # c2 = y - r
        # d(c2_i)/dy_i = 1, d(c2_i)/dr_i = -1
        for i in range(n):
            J[n+i, 3*i+1] = 1.0
            J[n+i, 3*i+2] = -1.0
        # c3 = w - x - r
        # d(c3_i)/dx_i = -1, d(c3_i)/dr_i = -1
        for i in range(n):
            J[2*n+i, 3*i] = -1.0
            J[2*n+i, 3*i+2] = -1.0
        # c4 = h - y - r
        # d(c4_i)/dy_i = -1, d(c4_i)/dr_i = -1
        for i in range(n):
            J[3*n+i, 3*i+1] = -1.0
            J[3*n+i, 3*i+2] = -1.0
        return J

    def overlap_constraints(params):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                # dist - (r_i + r_j) >= 0
                cons.append(dist - r[i] - r[j])
        return np.array(cons)
    
    def overlap_jac(params):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        J = np.zeros((n*(n-1)//2, 3*n))
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                if dist_sq < 1e-12:
                    dist = 1e-6
                    nx = 0
                    ny = 0
                else:
                    dist = np.sqrt(dist_sq)
                    nx = dx / dist
                    ny = dy / dist
                
                # d(dist - r_i - r_j)/dx_i = nx
                # d(dist - r_i - r_j)/dy_i = ny
                # d(dist - r_i - r_j)/dr_i = -1
                # d(dist - r_i - r_j)/dx_j = -nx
                # d(dist - r_i - r_j)/dy_j = -ny
                # d(dist - r_i - r_j)/dr_j = -1
                
                J[k, 3*i] = nx
                J[k, 3*i+1] = ny
                J[k, 3*i+2] = -1.0
                J[k, 3*j] = -nx
                J[k, 3*j+1] = -ny
                J[k, 3*j+2] = -1.0
                k += 1
        return J

    constraints = [
        {
            'type': 'ineq',
            'fun': boundary_constraints,
            'jac': boundary_jac
        },
        {
            'type': 'ineq',
            'fun': overlap_constraints,
            'jac': overlap_jac
        }
    ]
    return constraints

def _make_init_grid(w, h):
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cx = (i + 0.5) * w / side
            cy = (j + 0.5) * h / side
            r = 0.4 * min(w, h) / side
            cells.append([cx, cy, r])
    return np.array(cells[:N])

def _make_init_hex(w, h):
    # 5 rows: 3, 4, 5, 4, 3? Sum=19.
    # Let's try 4 rows: 5, 6, 6, 4? Sum=21.
    # Or 5 rows: 4, 5, 5, 5, 2?
    # Let's just use a dense grid for simplicity in init, SLSQP will move them.
    # Or a slightly better hex-like init.
    rows = [4, 5, 5, 5, 2] # Sum 21
    circles = []
    # Approximate scale
    scale = min(w, h) / 5.0
    dy = scale * np.sqrt(3) / 2
    total_h = 4 * dy
    offset_y = (h - total_h) / 2
    
    for i, n_circles in enumerate(rows):
        y = offset_y + i * dy
        # Center row
        row_width = (n_circles - 1) * scale
        start_x = (w - row_width) / 2
        for j in range(n_circles):
            x = start_x + j * scale
            r = 0.45 * scale
            circles.append([x, y, r])
    return np.array(circles[:N])

def _make_init_random(w, h, seed):
    rng = np.random.default_rng(seed)
    circles = []
    # Start with small radii to avoid immediate massive overlaps
    r_max = min(w, h) / 4.0
    for _ in range(N):
        r = rng.uniform(0.05 * r_max, 0.15 * r_max)
        x = rng.uniform(r, w - r)
        y = rng.uniform(r, h - r)
        circles.append([x, y, r])
    return np.array(circles)

def _solve_packing(w, h, init_circles, deadline):
    """
    Solve packing problem for given w, h and initial guess using SLSQP.
    """
    params0 = init_circles.flatten()
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0, w))      # x_i
        bounds.append((0, h))      # y_i
        bounds.append((1e-6, min(w, h)/2)) # r_i

    constraints = _make_constraints(w, h)
    
    # SLSQP options
    options = {
        'maxiter': 150,
        'ftol': 1e-8,
        'disp': False
    }

    try:
        res = minimize(
            _objective,
            params0,
            jac=_jac_objective,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options=options
        )
        
        if res.success or (hasattr(res, 'x') and res.x is not None):
            circles = res.x.reshape(N, 3)
            # Basic cleanup
            circles[:, 2] = np.maximum(circles[:, 2], 1e-6)
            
            # Validate and return
            if _validate(circles, w, h):
                return circles, _total_radius(circles)
            
            # If invalid, try to repair slightly
            # Simple repair: shrink radii if overlapping
            # This is a fallback, SLSQP should be valid
            return None, 0.0
        else:
            return None, 0.0
            
    except Exception:
        return None, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_value = 0.0
    best_result = None
    
    # Aspect ratios to try
    # w + h = 2
    # w = ar * h => h(ar+1) = 2 => h = 2/(ar+1), w = 2ar/(ar+1)
    ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45, 1.1, 0.9, 1.3, 0.75, 1.5, 0.65, 2.0, 0.55]
    
    # Initializations to try for each aspect ratio
    inits = ['grid', 'hex', 'rand1', 'rand2']
    
    for ar in ratios:
        if time.time() > deadline:
            break
            
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        # Also try swapped w, h if ar != 1
        if abs(ar - 1.0) > 1e-6:
            w2, h2 = h, w
        else:
            w2, h2 = None, None
            
        pairs = [(w, h)]
        if w2 is not None:
            pairs.append((w2, h2))
            
        for ww, hh in pairs:
            if time.time() > deadline:
                break
                
            for init_type in inits:
                if time.time() > deadline:
                    break
                    
                if init_type == 'grid':
                    init_circles = _make_init_grid(ww, hh)
                elif init_type == 'hex':
                    init_circles = _make_init_hex(ww, hh)
                elif init_type == 'rand1':
                    init_circles = _make_init_random(ww, hh, seed=42)
                elif init_type == 'rand2':
                    init_circles = _make_init_random(ww, hh, seed=123)
                else:
                    continue
                    
                circles, val = _solve_packing(ww, hh, init_circles, deadline)
                
                if circles is not None and val > best_value:
                    best_value = val
                    best_result = circles.copy()
                    
    # Fallback if no valid result found (should not happen)
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * 1.0 / side
                y = (j + 0.5) * 1.0 / side
                r = 0.4 / side
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result
