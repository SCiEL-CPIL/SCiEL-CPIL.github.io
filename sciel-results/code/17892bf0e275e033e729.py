import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _validate(x, y, r, W, H, tol=1e-6):
    """Check if circles are valid (within bounds, non-overlapping)."""
    if np.any(r <= 0):
        return False
    if np.any(x < r - tol) or np.any(x > W - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > H - r + tol):
        return False
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _penalty(x, y, r, W, H, beta=20.0):
    """Compute penalty for constraint violations."""
    n = len(x)
    pen = 0.0
    # Boundary violations
    viol_x_lo = np.maximum(0, r - x)
    viol_x_hi = np.maximum(0, x - (W - r))
    viol_y_lo = np.maximum(0, r - y)
    viol_y_hi = np.maximum(0, y - (H - r))
    pen += np.sum(viol_x_lo**2) + np.sum(viol_x_hi**2) + np.sum(viol_y_lo**2) + np.sum(viol_y_hi**2)
    
    # Overlap violations
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            viol = np.maximum(0, r[i] + r[j] - dist)
            pen += viol**2
    return pen

def _solve_single(W, H, x0, y0, r0, timeout=30.0):
    """Solve for a given rectangle and initial guess."""
    start = time.time()
    n = N
    
    def objective(v):
        if time.time() - start > timeout:
            return 1e10
        xi, yi, ri = v[:n], v[n:2*n], v[2*n:3*n]
        obj = -np.sum(ri)
        pen = _penalty(xi, yi, ri, W, H)
        return obj + 50.0 * pen
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-6, W - 1e-6))  # x_i
    for i in range(n):
        bounds.append((1e-6, H - 1e-6))  # y_i
    for i in range(n):
        bounds.append((1e-6, min(W, H) / 2))  # r_i
    
    v0 = np.concatenate([x0, y0, r0])
    
    try:
        res = minimize(objective, v0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 500, 'maxfun': 1000, 'disp': False})
        if res.success or res.fun < objective(v0):
            xi, yi, ri = res.x[:n], res.x[n:2*n], res.x[2*n:3*n]
            if _validate(xi, yi, ri, W, H):
                return xi, yi, ri, np.sum(ri)
    except Exception:
        pass
    
    return None

def _grid_init(W, H, m, n):
    """Create grid initialization."""
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
    r_est = min(W/(2*m) if m > 1 else W/2, H/(2*n) if n > 1 else H/2) * 0.9
    r0 = np.full(N, r_est)
    return x0, y0, r0

def _hex_init(W, H):
    """Create hexagonal packing initialization."""
    rows = int(np.ceil(np.sqrt(N)))
    r_est = min(W / (rows + 1), H / (rows * np.sqrt(3) + 1)) * 0.85
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

def _random_init(W, H):
    """Random initialization."""
    x0 = np.random.uniform(0.1, W - 0.1, N)
    y0 = np.random.uniform(0.1, H - 0.1, N)
    r0 = np.full(N, 0.05)
    return x0, y0, r0

def construct():
    start_time = time.time()
    deadline = 145  # 5s margin
    
    best_value = -np.inf
    best_result = None
    
    # Try multiple aspect ratios
    aspects = [1.0, 0.9, 1.1, 0.8, 1.25, 0.85, 1.15, 1.333, 0.75, 1.5, 0.667]
    
    for aspect in aspects:
        if time.time() > deadline:
            break
        
        W = 2.0 * aspect / (1 + aspect)
        H = 2.0 - W
        
        # Try different layouts
        layouts = []
        
        # Grid layouts
        for m, n in [(5,5), (4,6), (6,4), (3,7), (7,3), (2,11), (11,2), (5,4), (4,5)]:
            layout = _grid_init(W, H, m, n)
            if layout is not None:
                layouts.append(layout)
        
        # Hex layout
        hexl = _hex_init(W, H)
        if hexl is not None:
            layouts.append(hexl)
        
        # Random layouts
        for _ in range(3):
            layouts.append(_random_init(W, H))
        
        for x0, y0, r0 in layouts:
            if time.time() > deadline:
                break
            
            # Solve
            result = _solve_single(W, H, x0, y0, r0, timeout=20.0)
            if result is not None:
                xi, yi, ri, value = result
                if value > best_value:
                    best_value = value
                    best_result = np.column_stack([xi, yi, ri])
    
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
