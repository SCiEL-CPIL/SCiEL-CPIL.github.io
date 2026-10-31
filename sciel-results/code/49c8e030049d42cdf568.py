import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
PERIMETER = 4.0
DEADLINE_FRACTION = 0.90  # Safety margin against 120s/150s limit

def construct():
    t0 = time.time()
    deadline = 120.0 * DEADLINE_FRACTION
    
    best_val = -1.0
    best_circles = None
    
    # Aspect ratios to try: W + H = 2. 
    # We sample a range around the square (1.0, 1.0) and some elongated shapes.
    # W is width, H is height.
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.45]
    
    # Shuffle to vary the order of attempts, potentially finding a good solution earlier
    # if the "best" aspect ratio isn't the first one.
    random.shuffle(aspect_ratios)
    
    for W in aspect_ratios:
        if time.time() - t0 > deadline:
            break
        H = 2.0 - W
        if H <= 0.1 or W <= 0.1:
            continue
        
        # Try a few different initial grid configurations for this aspect ratio
        # Grid dimensions that fit 21 circles reasonably well
        grid_configs = [
            (5, 5), # 25 cells, use 21
            (4, 6), # 24 cells, use 21
            (3, 7), # 21 cells exactly
            (7, 3), # 21 cells exactly
            (6, 4), # 24 cells
            (5, 4), # 20 cells, add 1? No, stick to >=21 cells or handle remainder
        ]
        
        # Filter valid grid configs where rows*cols >= N
        valid_configs = [(r, c) for r, c in grid_configs if r * c >= N]
        
        for rows, cols in valid_configs:
            if time.time() - t0 > deadline:
                break
            
            circles = _optimize_aspect(W, H, rows, cols, t0, deadline)
            if circles is None:
                continue
            
            # Scale to perimeter 4
            minx = min(c[0] - c[2] for c in circles)
            miny = min(c[1] - c[2] for c in circles)
            maxx = max(c[0] + c[2] for c in circles)
            maxy = max(c[1] + c[2] for c in circles)
            
            bw = maxx - minx
            bh = maxy - miny
            per = 2.0 * (bw + bh)
            
            if per <= 0:
                continue
                
            s = 4.0 / per
            scaled_circles = []
            for (x, y, r) in circles:
                nx = (x - minx) * s
                ny = (y - miny) * s
                nr = r * s
                scaled_circles.append((nx, ny, nr))
            
            # Validate
            if _is_valid(scaled_circles, 2.0, 2.0):
                val = sum(c[2] for c in scaled_circles)
                if val > best_val:
                    best_val = val
                    best_circles = scaled_circles
    
    if best_circles is None:
        # Fallback: simple grid
        side = 5
        pts = []
        for i in range(side):
            for j in range(side):
                if len(pts) < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 0.01
                    pts.append((x, y, r))
        best_circles = pts
        minx = min(c[0] - c[2] for c in best_circles)
        miny = min(c[1] - c[2] for c in best_circles)
        maxx = max(c[0] + c[2] for c in best_circles)
        maxy = max(c[1] + c[2] for c in best_circles)
        bw = maxx - minx
        bh = maxy - miny
        per = 2.0 * (bw + bh)
        s = 4.0 / per
        best_circles = [((x-minx)*s, (y-miny)*s, r*s) for (x,y,r) in best_circles]
        
    return np.array(best_circles)

def _is_valid(circles, W, H, tol=1e-6):
    n = len(circles)
    if n != N:
        return False
    # Check bounds
    for x, y, r in circles:
        if x - r < -tol or x + r > W + tol or y - r < -tol or y + r > H + tol:
            return False
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, r1 = circles[i]
            x2, y2, r2 = circles[j]
            d = np.hypot(x1-x2, y1-y2)
            if d < r1 + r2 - tol:
                return False
    return True

def _objective(x, W, H, mu=100.0):
    """
    Objective function for minimization.
    We want to maximize sum(r), so we minimize -sum(r) + penalties.
    x = [x1, y1, r1, x2, y2, r2, ...]
    """
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    val = -np.sum(r)
    
    # Wall violations penalty
    for i in range(N):
        if cx[i] - r[i] < 0:
            val += mu * (cx[i] - r[i]) ** 2
        if cy[i] - r[i] < 0:
            val += mu * (cy[i] - r[i]) ** 2
        if cx[i] + r[i] > W:
            val += mu * (cx[i] + r[i] - W) ** 2
        if cy[i] + r[i] > H:
            val += mu * (cy[i] + r[i] - H) ** 2
    
    # Overlap violations penalty
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.sqrt(dx * dx + dy * dy)
            min_d = r[i] + r[j]
            if d < min_d:
                val += mu * (min_d - d) ** 2
    
    return val

def _optimize_aspect(W, H, rows, cols, t0, deadline):
    """
    Optimizes circle packing for a specific aspect ratio using SLSQP.
    """
    # Initialize on a grid
    cx = np.zeros(N)
    cy = np.zeros(N)
    r = np.zeros(N)
    
    count = 0
    for i in range(cols):
        for j in range(rows):
            if count >= N:
                break
            cx[count] = (i + 0.5) * (W / cols)
            cy[count] = (j + 0.5) * (H / rows)
            # Initial radius slightly smaller than cell half-size
            r[count] = 0.9 * min(W / cols, H / rows) / 2.0
            count += 1
            if count >= N:
                break
        if count >= N:
            break
            
    # Ensure we have N circles
    if count < N:
        # This shouldn't happen if rows*cols >= N, but just in case
        return None

    x0 = np.concatenate([cx, cy, r])
    
    # Bounds:
    # x in [0, W], y in [0, H], r in [1e-6, min(W,H)/2]
    bounds = []
    for i in range(N):
        bounds.append((0, W)) # x
    for i in range(N):
        bounds.append((0, H)) # y
    for i in range(N):
        bounds.append((1e-6, min(W, H) / 2.0)) # r
        
    # SLSQP is good for constrained problems, but can be slow.
    # We limit iterations and check time.
    max_iter = 200
    
    try:
        res = minimize(_objective, x0, args=(W, H, 100.0), method='SLSQP',
                       bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-8})
        
        if res.success or res.fun < _objective(x0, W, H, 100.0):
            cx_opt = res.x[0::3]
            cy_opt = res.x[1::3]
            r_opt = res.x[2::3]
            
            circles = list(zip(cx_opt, cy_opt, r_opt))
            
            # Local refinement: try to increase radii by nudging positions
            circles = _refine(circles, W, H, t0, deadline)
            
            return circles
    except Exception:
        pass
        
    return None

def _refine(circles, W, H, t0, deadline, steps=500, lr0=0.05):
    """
    Local search: nudge each circle to increase its radius while keeping constraints.
    """
    circles = [list(c) for c in circles]
    n = len(circles)
    rng = random.Random(42)
    
    for step in range(steps):
        if time.time() - t0 > deadline:
            break
            
        lr = lr0 * (1.0 - step / steps)
        i = rng.randrange(n)
        x, y, r = circles[i]
        
        # Try a few random directions
        improved = False
        for _ in range(5):
            dx = rng.uniform(-1, 1)
            dy = rng.uniform(-1, 1)
            norm = np.hypot(dx, dy)
            if norm < 1e-9:
                continue
            dx, dy = dx / norm, dy / norm
            nx = x + dx * lr
            ny = y + dy * lr
            
            # Check bounds for new position
            if nx < 0 or nx > W or ny < 0 or ny > H:
                continue
            
            # Calculate max radius at new position
            nr = min(nx, W - nx, ny, H - ny)
            for j in range(n):
                if j == i:
                    continue
                px, py, pr = circles[j]
                d = np.hypot(nx - px, ny - py)
                cand = d - pr
                if cand < nr:
                    nr = cand
            
            nr = max(nr, 0.0)
            
            # If we can make the circle bigger, accept the move
            if nr > r + 1e-12:
                circles[i] = [nx, ny, nr]
                improved = True
                break
                
    return circles

if __name__ == "__main__":
    res = construct()
    print(res.shape)
    print(res[:3])
