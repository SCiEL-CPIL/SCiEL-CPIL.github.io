import numpy as np
import time

N = 21
from scipy.optimize import minimize

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

def _smooth_min_penalty(x, y, r, W, H, beta=50.0):
    """Compute smooth penalty for constraint violations using log-sum-exp style soft-min."""
    n = len(x)
    # Boundary violations: want x >= r, x <= W-r, y >= r, y <= H-r
    # Penalty terms are max(0, violation)
    pen = 0.0
    # x lower: r - x should be <= 0
    viol_x_lo = np.maximum(0, r - x)
    viol_x_hi = np.maximum(0, x - (W - r))
    viol_y_lo = np.maximum(0, r - y)
    viol_y_hi = np.maximum(0, y - (H - r))
    
    # Overlap violations: r_i + r_j - dist should be <= 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            viol = np.maximum(0, r[i] + r[j] - dist)
            pen += viol**2
    
    pen += np.sum(viol_x_lo**2) + np.sum(viol_x_hi**2) + np.sum(viol_y_lo**2) + np.sum(viol_y_hi**2)
    return pen

def _relax(x, y, r, W, H, iters=2000, tol=1e-9):
    """Iterative relaxation: expand radii and resolve overlaps."""
    n = len(x)
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for it in range(iters):
        # Try to expand all radii slightly
        growth = 0.001
        r_new = r + growth
        
        # Check if expansion is valid
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
            # Resolve overlaps by pushing circles apart
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist - tol:
                        # Push apart
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
            
            # Clamp to boundaries
            for i in range(n):
                x[i] = np.clip(x[i], r[i], W - r[i])
                y[i] = np.clip(y[i], r[i], H - r[i])
            
            if not moved:
                # No overlaps to resolve, try smaller growth
                growth *= 0.5
                if growth < 1e-10:
                    break
    
    return x, y, r

def _scipy_polish(x, y, r, W, H, timeout=5.0):
    """Use scipy minimize to polish the packing by maximizing sum of radii."""
    start = time.time()
    n = len(x)
    
    def objective(v):
        if time.time() - start > timeout:
            return 1e10
        xi, yi, ri = v[:n], v[n:2*n], v[2*n:3*n]
        # Negative sum of radii (we minimize)
        obj = -np.sum(ri)
        # Add penalty for constraint violations
        pen = _smooth_min_penalty(xi, yi, ri, W, H)
        return obj + 100.0 * pen
    
    # Bounds for each variable
    bounds = []
    for i in range(n):
        bounds.append((1e-6, W - 1e-6))  # x_i
    for i in range(n):
        bounds.append((1e-6, H - 1e-6))  # y_i
    for i in range(n):
        bounds.append((1e-6, min(W, H) / 2))  # r_i
    
    v0 = np.concatenate([x, y, r])
    
    try:
        res = minimize(objective, v0, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 200, 'maxfun': 500, 'disp': False})
        if res.success or res.fun < objective(v0):
            xi, yi, ri = res.x[:n], res.x[n:2*n], res.x[2*n:3*n]
            if _validate(xi, yi, ri, W, H):
                return xi, yi, ri
    except Exception:
        pass
    
    return None

def _grid_layout(W, H, m, n):
    """Create a grid layout of circles."""
    if m * n < N:
        return None
    xs = np.linspace(W/(2*m), W - W/(2*m), m) if m > 1 else np.array([W/2])
    ys = np.linspace(H/(2*n), H - H/(2*n), n) if n > 1 else np.array([H/2])
    gx, gy = np.meshgrid(xs, ys)
    gx = gx.flatten()
    gy = gy.flatten()
    
    # Take first N
    idx = np.arange(N)
    x0 = gx[idx].copy()
    y0 = gy[idx].copy()
    r_est = min(W/(2*m) if m > 1 else W/2, H/(2*n) if n > 1 else H/2) * 0.95
    r0 = np.full(N, r_est)
    return x0, y0, r0

def _hex_layout(W, H):
    """Create a hexagonal packing layout."""
    # Estimate radius for hex packing
    # In hex packing, each circle has 6 neighbors
    # Row spacing = r * sqrt(3), column spacing = 2*r
    # For N circles, roughly sqrt(N) rows
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
            
            # Relax
            x_rel, y_rel, r_rel = _relax(x0, y0, r0, W, H, iters=3000)
            
            # Validate
            if _validate(x_rel, y_rel, r_rel, W, H):
                value = np.sum(r_rel)
                if value > best_value:
                    best_value = value
                    best_result = np.column_stack([x_rel, y_rel, r_rel])
                
                # Try scipy polish if time allows
                if time.time() < deadline - 10:
                    polished = _scipy_polish(x_rel, y_rel, r_rel, W, H, timeout=5.0)
                    if polished is not None:
                        px, py, pr = polished
                        pval = np.sum(pr)
                        if pval > best_value:
                            best_value = pval
                            best_result = np.column_stack([px, py, pr])
    
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
