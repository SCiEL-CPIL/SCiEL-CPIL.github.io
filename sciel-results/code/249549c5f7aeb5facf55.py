"""Circle packing in perimeter-4 rectangle using SLSQP with hard non-overlap constraints."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _hex_init(W, H, n, seed=0):
    """
    Generate a hexagonal grid initialization scaled to fit W x H.
    """
    rng = np.random.default_rng(seed)
    
    # Estimate radius to fit N circles in W x H
    # Density of hex packing ~ 0.9
    # N * pi * r^2 ~ 0.9 * W * H
    r_est = np.sqrt(0.9 * W * H / (N * np.pi))
    
    # Create a grid of points
    # We want to generate points in a normalized space and scale them.
    # Let's generate a hex grid with spacing 1.0
    cols = int(np.ceil(np.sqrt(n * W / H))) # Adjust cols based on aspect ratio
    if cols < 1: cols = 1
    rows = int(np.ceil(n / cols))
    
    xs = []
    ys = []
    for row in range(rows):
        for col in range(cols):
            if len(xs) >= n:
                break
            # Hex offset
            x = col + 0.5 * (row % 2)
            y = row * np.sqrt(3) / 2
            xs.append(x)
            ys.append(y)
            
    # If we didn't get enough points, fill with random
    while len(xs) < n:
        xs.append(rng.uniform(0, max(xs) if xs else 1))
        ys.append(rng.uniform(0, max(ys) if ys else 1))
        
    xs = np.array(xs[:n])
    ys = np.array(ys[:n])
    
    # Scale to fit W and H
    # Center the grid
    cx = (xs - np.mean(xs)) * (W / (np.max(xs) - np.min(xs) + 1e-6)) + W/2
    cy = (ys - np.mean(ys)) * (H / (np.max(ys) - np.min(ys) + 1e-6)) + H/2
    
    # Clamp to box
    cx = np.clip(cx, 0.01, W - 0.01)
    cy = np.clip(cy, 0.01, H - 0.01)
    
    # Initial radii: small to ensure no overlap
    # Calculate min distance
    min_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)
            min_dist = min(min_dist, d)
            
    r = np.full(n, min_dist * 0.4)
    
    # Jitter
    cx += rng.uniform(-0.02, 0.02, n)
    cy += rng.uniform(-0.02, 0.02, n)
    cx = np.clip(cx, 0.01, W - 0.01)
    cy = np.clip(cy, 0.01, H - 0.01)
    
    return cx, cy, r

def _is_valid(cx, cy, r, W, H, tol=1e-7):
    """
    Check if the packing is valid.
    """
    n = len(cx)
    # Wall constraints
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    # Pairwise non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist_sq = dx * dx + dy * dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist * min_dist - tol:
                return False
    return True

def _optimize_aspect(W, H, deadline, seed=0):
    """
    Optimize for a specific aspect ratio using SLSQP with hard non-overlap constraints.
    """
    if time.time() > deadline:
        return None, -float('inf')
        
    cx, cy, r = _hex_init(W, H, N, seed=seed)
    
    # Variables: x, y, r
    x0 = np.concatenate([cx, cy, r])
    n = N
    
    # Objective: Maximize sum(r) -> Minimize -sum(r)
    def objective(x):
        r = x[2*n:3*n]
        return -np.sum(r)
        
    # Constraints: Non-overlap
    # We use inequality constraints: dist^2 >= (ri + rj)^2
    # dist^2 - (ri + rj)^2 >= 0
    cons = []
    for i in range(n):
        for j in range(i + 1, n):
            def constraint(x, i=i, j=j):
                xi, yi = x[i], x[n+i]
                xj, yj = x[j], x[n+j]
                ri, rj = x[2*n+i], x[2*n+j]
                dx = xi - xj
                dy = yi - yj
                return dx*dx + dy*dy - (ri + rj)**2
            cons.append({'type': 'ineq', 'fun': constraint})
            
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0, W)) # x
    for i in range(n):
        bounds.append((0, H)) # y
    for i in range(n):
        bounds.append((1e-4, min(W, H) / 2)) # r

    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                       constraints=cons, options={'maxiter': 100, 'ftol': 1e-8})
        
        if res.success or res.fun < 0: # res.fun is -sum(r), so we want it to be negative (large sum)
            cx_opt = res.x[:n]
            cy_opt = res.x[n:2*n]
            r_opt = res.x[2*n:3*n]
            
            # Validate
            if _is_valid(cx_opt, cy_opt, r_opt, W, H):
                return (cx_opt, cy_opt, r_opt), np.sum(r_opt)
                
            # If slightly invalid, try to fix with a small penalty step or just reject
            # For now, reject if invalid.
            
    except Exception:
        pass
        
    return None, -float('inf')

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_val = -float('inf')
    best_config = None
    
    # Try different aspect ratios
    # Perimeter 4 => W + H = 2
    # Square (1,1) is a good start.
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0]
    
    np.random.seed(42)
    
    for W in aspect_ratios:
        if time.time() > deadline:
            break
        H = 2.0 - W
        if H <= 0.1:
            continue
            
        # Perform a few restarts
        # SLSQP with constraints is slow. 1 restart per aspect ratio might be enough if init is good.
        # Let's do 2 restarts if time permits.
        num_restarts = 2 if time.time() < deadline - 20 else 1
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            config, val = _optimize_aspect(W, H, deadline, seed=restart)
            
            if val > best_val:
                best_val = val
                best_config = config
                
    if best_config is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    cx, cy, r = best_config
    return np.column_stack([cx, cy, r])
