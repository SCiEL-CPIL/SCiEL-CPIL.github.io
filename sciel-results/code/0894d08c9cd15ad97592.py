import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _validate(x, y, r, W, H, tol=1e-6):
    """Check if configuration is valid."""
    if np.any(r <= 0):
        return False
    if np.any(x < r - tol) or np.any(x > W - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > H - r + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _objective(v, W, H):
    """Objective: minimize negative sum of radii + penalties."""
    x = v[0:N]
    y = v[N:2*N]
    r = v[2*N:3*N]
    
    val = -np.sum(r)
    
    # Boundary penalties
    pen = 0.0
    pen += np.sum(np.maximum(0, r - x)**2)
    pen += np.sum(np.maximum(0, x - (W - r))**2)
    pen += np.sum(np.maximum(0, r - y)**2)
    pen += np.sum(np.maximum(0, y - (H - r))**2)
    
    # Pairwise overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            req = r[i] + r[j]
            if dist_sq < req * req:
                dist = np.sqrt(dist_sq)
                pen += (req - dist)**2
    
    return val + 1000.0 * pen

def _hex_init(W, H, n_circles):
    """Create hexagonal grid initialization."""
    xs = np.zeros(n_circles)
    ys = np.zeros(n_circles)
    rs = np.zeros(n_circles)
    
    # Hexagonal packing: rows offset
    # Estimate spacing
    area_per_circle = W * H / n_circles
    spacing = np.sqrt(area_per_circle)
    
    row_spacing = spacing * np.sqrt(3) / 2
    col_spacing = spacing
    
    # Try to fit as many as possible in hex pattern
    k = 0
    row = 0
    while k < n_circles:
        # Alternate rows offset
        offset = (col_spacing / 2) if (row % 2 == 1) else 0
        col = 0
        while k < n_circles:
            cx = offset + col * col_spacing + col_spacing / 2
            cy = row * row_spacing + row_spacing / 2
            if cx < W and cy < H:
                r = 0.45 * min(col_spacing, row_spacing)
                if r > 0 and cx - r > 0 and cx + r < W and cy - r > 0 and cy + r < H:
                    xs[k] = cx
                    ys[k] = cy
                    rs[k] = r
                    k += 1
            col += 1
        row += 1
        if row > 50:  # safety
            break
    
    return xs, ys, rs

def _grid_init(W, H, n_circles):
    """Create grid initialization."""
    xs = np.zeros(n_circles)
    ys = np.zeros(n_circles)
    rs = np.zeros(n_circles)
    
    cols = int(np.sqrt(n_circles))
    rows = int(np.ceil(n_circles / cols))
    
    k = 0
    for i in range(cols):
        for j in range(rows):
            if k >= n_circles:
                break
            cx = (i + 0.5) * W / cols
            cy = (j + 0.5) * H / rows
            r = 0.45 * min(W / cols, H / rows)
            xs[k] = cx
            ys[k] = cy
            rs[k] = r
            k += 1
    
    return xs, ys, rs

def _optimize_aspect(W, H, init_x, init_y, init_r, deadline):
    """Optimize for a given aspect ratio."""
    x0 = np.concatenate([init_x, init_y, init_r])
    
    max_r_global = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((0, W))
        bounds.append((0, H))
        bounds.append((1e-8, max_r_global))
    
    # First pass: L-BFGS-B
    try:
        res = minimize(_objective, x0, args=(W, H), method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10})
    except Exception:
        return None
    
    # Check time
    if time.time() > deadline:
        return None
    
    # Second pass: polish with tighter tolerance
    x = res.x[0:N]
    y = res.x[N:2*N]
    r = res.x[2*N:3*N]
    
    # Validate
    if not _validate(x, y, r, W, H, tol=1e-5):
        return None
    
    # Polish
    try:
        res2 = minimize(_objective, res.x, args=(W, H), method='L-BFGS-B',
                        bounds=bounds, options={'maxiter': 300, 'ftol': 1e-12})
    except Exception:
        return None
    
    x = res2.x[0:N]
    y = res2.x[N:2*N]
    r = res2.x[2*N:3*N]
    
    if not _validate(x, y, r, W, H, tol=1e-5):
        return None
    
    return np.column_stack((x, y, r)), np.sum(r)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_val = -1
    best_res = None
    
    # Try different aspect ratios
    widths = np.linspace(0.65, 1.35, 15)
    
    for W in widths:
        if time.time() > deadline - 10:
            break
        
        H = 2.0 - W
        if H <= 0.1:
            continue
        
        # Try both hex and grid init
        for init_func in [_hex_init, _grid_init]:
            if time.time() > deadline - 5:
                break
            
            init_x, init_y, init_r = init_func(W, H, N)
            
            result = _optimize_aspect(W, H, init_x, init_y, init_r, deadline)
            
            if result is not None:
                res_arr, val = result
                if val > best_val:
                    best_val = val
                    best_res = res_arr
    
    if best_res is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3])
        return np.array(cells[:N])
    
    return best_res
