"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_start_time = time.time()
TIME_LIMIT = 140  # seconds, leave margin for 150s kill

def _elapsed():
    return time.time() - _start_time

def _penalty_and_grad(params, w, h):
    """
    params: flat array of length 3*N: [x1,y1,r1, x2,y2,r2, ...]
    Minimize: -sum(r_i) + C * (overlap penalties + boundary penalties)
    """
    n = len(params) // 3
    pts = params.reshape(n, 3)
    xs = pts[:, 0]
    ys = pts[:, 1]
    rs = pts[:, 2]
    
    # Objective: -sum(r)
    obj = -np.sum(rs)
    
    # Overlap penalties
    C = 10.0
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                overlap_pen += (min_dist - dist)**2
    
    # Boundary penalties
    boundary_pen = 0.0
    for i in range(n):
        if xs[i] < rs[i]:
            boundary_pen += (rs[i] - xs[i])**2
        if xs[i] > w - rs[i]:
            boundary_pen += (xs[i] - (w - rs[i]))**2
        if ys[i] < rs[i]:
            boundary_pen += (rs[i] - ys[i])**2
        if ys[i] > h - rs[i]:
            boundary_pen += (ys[i] - (h - rs[i]))**2
    
    total = obj + C * (overlap_pen + boundary_pen)
    return total

def _init_grid(w, h, n_circles, rows, cols, jitter=0.0):
    """Initialize circles in a grid pattern."""
    circles = []
    for i in range(rows):
        for j in range(cols):
            if len(circles) >= n_circles:
                break
            x = (j + 0.5) * w / cols
            y = (i + 0.5) * h / rows
            # Estimate radius based on cell size
            cell_w = w / cols
            cell_h = h / rows
            r = min(cell_w, cell_h) * 0.45
            if jitter > 0:
                x += (np.random.rand() - 0.5) * jitter * cell_w
                y += (np.random.rand() - 0.5) * jitter * cell_h
            circles.append([x, y, r])
    return np.array(circles)

def _optimize_for_aspect(w, h, seed=0):
    """Try to optimize packing for given aspect ratio."""
    np.random.seed(seed)
    
    # Try different grid configurations
    best_val = -np.inf
    best_params = None
    
    configs = []
    # Various row/col combinations for 21 circles
    for rows in range(3, 8):
        for cols in range(3, 8):
            if rows * cols >= 21:
                configs.append((rows, cols))
    
    # Also try some hexagonal-like offsets
    for rows, cols in configs:
        if _elapsed() > TIME_LIMIT:
            break
        for jitter in [0.0, 0.1, 0.2]:
            if _elapsed() > TIME_LIMIT:
                break
            circles = _init_grid(w, h, 21, rows, cols, jitter=jitter)
            if len(circles) < 21:
                continue
            params = circles.flatten()
            
            # Bounds
            bounds = []
            for i in range(21):
                bounds.append((0, w))  # x
                bounds.append((0, h))  # y
                bounds.append((0.01, min(w, h)/2))  # r
            
            try:
                result = minimize(_penalty_and_grad, params, args=(w, h),
                                method='L-BFGS-B', bounds=bounds,
                                options={'maxiter': 200, 'maxfun': 5000})
            except Exception:
                continue
            
            if _elapsed() > TIME_LIMIT:
                break
            
            # Check validity
            pts = result.x.reshape(21, 3)
            valid = True
            for i in range(21):
                x, y, r = pts[i]
                if x < r or x > w - r or y < r or y > h - r:
                    valid = False
                    break
            if valid:
                for i in range(21):
                    for j in range(i+1, 21):
                        dx = pts[i,0] - pts[j,0]
                        dy = pts[i,1] - pts[j,1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist < pts[i,2] + pts[j,2] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                val = np.sum(pts[:, 2])
                if val > best_val:
                    best_val = val
                    best_params = result.x
    
    return best_val, best_params

def construct():
    global _start_time
    _start_time = time.time()
    
    best_val = -np.inf
    best_result = None
    
    # Try various aspect ratios
    aspect_ratios = np.linspace(0.3, 1/0.3, 15)  # w/h from 0.3 to ~3.3
    # Actually, w + h = 2, so w in [0.5, 1.5] for reasonable rectangles
    # Let's try w from 0.4 to 1.6
    w_values = np.linspace(0.4, 1.6, 12)
    
    for w in w_values:
        if _elapsed() > TIME_LIMIT:
            break
        h = 2.0 - w
        
        # Try a few seeds
        for seed in [0, 1, 2]:
            if _elapsed() > TIME_LIMIT:
                break
            val, params = _optimize_for_aspect(w, h, seed=seed)
            if val > best_val:
                best_val = val
                best_result = (w, h, params)
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    w, h, params = best_result
    pts = params.reshape(21, 3)
    
    # Final validation and cleanup
    # Ensure circles are within bounds and non-overlapping
    # If not, reduce radii slightly
    for i in range(21):
        x, y, r = pts[i]
        # Clamp to bounds
        r = min(r, x, w - x, y, h - y)
        r = max(r, 0.01)
        pts[i] = [x, y, r]
    
    # Check for overlaps and fix by reducing radii
    for _ in range(10):
        overlap = False
        for i in range(21):
            for j in range(i+1, 21):
                dx = pts[i,0] - pts[j,0]
                dy = pts[i,1] - pts[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = pts[i,2] + pts[j,2]
                if dist < min_dist:
                    # Reduce both radii
                    reduce = (min_dist - dist) / 2 + 0.001
                    pts[i,2] = max(pts[i,2] - reduce, 0.01)
                    pts[j,2] = max(pts[j,2] - reduce, 0.01)
                    overlap = True
        if not overlap:
            break
    
    # Final boundary check
    for i in range(21):
        x, y, r = pts[i]
        r = min(r, x, w - x, y, h - y)
        r = max(r, 0.01)
        pts[i,2] = r
    
    return pts
# EVOLVE-BLOCK-END
