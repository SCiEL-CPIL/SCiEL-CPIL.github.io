import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _validate(x, y, r, W, H, tol=1e-7):
    """Check if circles are valid (within bounds, non-overlapping)."""
    if np.any(r <= 0):
        return False
    if np.any(x < r - tol) or np.any(x > W - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > H - r + tol):
        return False
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _relax(x, y, r, W, H, iters=1500, tol=1e-9):
    """Iterative relaxation: expand radii and resolve overlaps."""
    n = len(x)
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for it in range(iters):
        growth = 0.001
        r_new = r + growth
        
        valid = True
        for i in range(n):
            if r_new[i] > x[i] or r_new[i] > W - x[i] or r_new[i] > y[i] or r_new[i] > H - y[i]:
                valid = False
                break
        if valid:
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r_new[i] + r_new[j] - tol:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            r = r_new
        else:
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist - tol:
                        if dist < 1e-12:
                            dx = 0.001
                            dy = 0.001
                            dist = np.sqrt(dx*dx + dy*dy)
                        overlap = min_dist - dist
                        fx = dx / dist * overlap * 0.5
                        fy = dy / dist * overlap * 0.5
                        x[i] += fx
                        y[i] += fy
                        x[j] -= fx
                        y[j] -= fy
                        moved = True
            
            for i in range(n):
                x[i] = np.clip(x[i], r[i], W - r[i])
                y[i] = np.clip(y[i], r[i], H - r[i])
            
            if not moved:
                growth *= 0.5
                if growth < 1e-10:
                    break
    
    return x, y, r

def _grid_layout(W, H, m, n):
    """Create a grid layout of circles."""
    if m * n < N:
        return None
    xs = np.linspace(W/(2*m), W - W/(2*m), m) if m > 1 else np.array([W/2])
    ys = np.linspace(H/(2*n), H - H/(2*n), n) if n > 1 else np.array([H/2])
    gx, gy = np.meshgrid(xs, ys)
    gx = gx.flatten()
    gy = gy.flatten()
    
    idx = np.arange(N)
    x0 = gx[idx].copy()
    y0 = gy[idx].copy()
    r_est = min(W/(2*m) if m > 1 else W/2, H/(2*n) if n > 1 else H/2) * 0.95
    r0 = np.full(N, r_est)
    return x0, y0, r0

def _hex_layout(W, H):
    """Create a hexagonal packing layout."""
    rows = int(np.ceil(np.sqrt(N)))
    r_est = min(W / (rows + 1), H / (rows * np.sqrt(3) + 1)) * 0.9
    r_est = min(r_est, 0.15)
    
    x0 = []
    y0 = []
    row_spacing = r_est * np.sqrt(3)
    col_spacing = 2 * r_est
    row = 0
    while len(x0) < N:
        cols_in_row = int(np.floor((W - 2*r_est) / col_spacing)) + 1
        offset = r_est if row % 2 == 1 else 0
        for c in range(cols_in_row):
            if len(x0) >= N:
                break
            cx = r_est + offset + c * col_spacing
            cy = r_est + row * row_spacing
            if cx + r_est <= W and cy + r_est <= H:
                x0.append(cx)
                y0.append(cy)
        row += 1
    
    if len(x0) < N:
        return None
    
    return np.array(x0[:N]), np.array(y0[:N]), np.full(N, r_est)

def _slsqp_solve(W, H, x0, y0, r0, maxiter=150):
    """Solve NLP with SLSQP for given aspect ratio."""
    n = N
    # Decision variables: x1..xn, y1..yn, r1..rn
    x_init = np.concatenate([x0, y0, r0])
    
    def objective(v):
        return -np.sum(v[2*n:])
    
    def jac(v):
        g = np.zeros(3*n)
        g[2*n:] = -1.0
        return g
    
    # Precompute pairs for overlap constraints
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def con_overlap(v):
        pos = v[:2*n].reshape(n, 2)
        r = v[2*n:]
        vals = np.empty(len(pairs))
        for idx, (i, j) in enumerate(pairs):
            d2 = (pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2
            s = r[i] + r[j]
            vals[idx] = d2 - s*s
        return vals
    
    def con_left(v):
        return v[:2*n:2] - v[2*n:]
    
    def con_right(v):
        return (W - v[:2*n:2]) - v[2*n:]
    
    def con_bottom(v):
        return v[1:2*n:2] - v[2*n:]
    
    def con_top(v):
        return (H - v[1:2*n:2]) - v[2*n:]
    
    cons = [
        {"type": "ineq", "fun": con_overlap},
        {"type": "ineq", "fun": con_left},
        {"type": "ineq", "fun": con_right},
        {"type": "ineq", "fun": con_bottom},
        {"type": "ineq", "fun": con_top},
    ]
    
    bounds = [(0.0, W)]*n + [(0.0, H)]*n + [(1e-4, 1.0)]*n
    
    try:
        res = minimize(objective, x_init, jac=jac, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': 1e-10})
    except Exception:
        return None
    
    if res.fun < 0:
        x_res = res.x[:n]
        y_res = res.x[n:2*n]
        r_res = res.x[2*n:]
        # Clamp to ensure validity
        x_res = np.clip(x_res, r_res, W - r_res)
        y_res = np.clip(y_res, r_res, H - r_res)
        if _validate(x_res, y_res, r_res, W, H):
            return x_res, y_res, r_res
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    best_value = -np.inf
    best_result = None
    
    aspects = [1.0, 0.9, 1.1, 0.8, 1.25, 0.85, 1.15, 1.333, 0.75, 1.5, 0.667]
    
    for aspect in aspects:
        if time.time() > deadline:
            break
        
        W = 2.0 * aspect / (1 + aspect)
        H = 2.0 - W
        
        layouts = []
        
        # Grid layouts
        for m, n in [(5,5), (4,6), (6,4), (3,7), (7,3), (2,11), (11,2), (5,4), (4,5)]:
            layout = _grid_layout(W, H, m, n)
            if layout is not None:
                layouts.append(layout)
        
        # Hex layout
        hexl = _hex_layout(W, H)
        if hexl is not None:
            layouts.append(hexl)
        
        for x0, y0, r0 in layouts:
            if time.time() > deadline:
                break
            
            # First try relaxation (fast)
            x_rel, y_rel, r_rel = _relax(x0, y0, r0, W, H, iters=1500)
            
            if _validate(x_rel, y_rel, r_rel, W, H):
                value = np.sum(r_rel)
                if value > best_value:
                    best_value = value
                    best_result = np.column_stack([x_rel, y_rel, r_rel])
            
            # Then try SLSQP for refinement (slower but potentially better)
            if time.time() < deadline - 10:
                slsqp_res = _slsqp_solve(W, H, x0, y0, r0, maxiter=100)
                if slsqp_res is not None:
                    x_s, y_s, r_s = slsqp_res
                    value_s = np.sum(r_s)
                    if value_s > best_value:
                        best_value = value_s
                        best_result = np.column_stack([x_s, y_s, r_s])
    
    if best_result is not None:
        return best_result
    else:
        # Fallback: simple grid in unit square
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4))
        return np.array(cells[:N])
