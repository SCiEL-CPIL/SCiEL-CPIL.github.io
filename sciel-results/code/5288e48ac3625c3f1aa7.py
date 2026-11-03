import numpy as np
import time
from scipy.optimize import minimize

N = 21
_DEADLINE = 150.0  # seconds
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _is_valid(circles, width, height, tol=1e-6):
    """Check circles are within [0,width]x[0,height] and non-overlapping."""
    for i in range(len(circles)):
        x, y, r = circles[i]
        if r < -tol:
            return False
        if x - r < -tol or x + r > width + tol:
            return False
        if y - r < -tol or y + r > height + tol:
            return False
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < ri + rj - tol:
                return False
    return True

def _penalty_objective(params, width, height):
    """
    params: flat array of length 3*N: [x0,y0,r0, x1,y1,r1, ...]
    Objective: minimize -sum(r_i) + penalty for constraint violations.
    Constraints:
      - r_i >= 0
      - x_i - r_i >= 0, x_i + r_i <= width
      - y_i - r_i >= 0, y_i + r_i <= height
      - dist(i,j) >= r_i + r_j
    """
    n = len(params) // 3
    penalty = 0.0
    obj = 0.0
    for i in range(n):
        x, y, r = params[3*i], params[3*i+1], params[3*i+2]
        obj -= r  # maximize sum of radii -> minimize negative
        # boundary penalties (soft)
        if x - r < 0:
            penalty += (x - r)**2
        if x + r > width:
            penalty += (x + r - width)**2
        if y - r < 0:
            penalty += (y - r)**2
        if y + r > height:
            penalty += (y + r - height)**2
        if r < 0:
            penalty += r**2
    # overlap penalties
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = params[3*i], params[3*i+1], params[3*i+2]
            xj, yj, rj = params[3*j], params[3*j+1], params[3*j+2]
            dx = xi - xj
            dy = yi - yj
            dist = np.sqrt(dx*dx + dy*dy + 1e-12)
            min_dist = ri + rj
            if dist < min_dist:
                penalty += (min_dist - dist)**2
    return obj + 100.0 * penalty

def _make_initial(width, height, seed=42):
    """Create initial positions: grid-like with small radii."""
    rng = np.random.RandomState(seed)
    # Use a grid to spread circles
    cols = 5
    rows = 5
    xs = np.linspace(0.1*width, 0.9*width, cols)
    ys = np.linspace(0.1*height, 0.9*height, rows)
    params = np.zeros(3*N)
    idx = 0
    for i in range(cols):
        for j in range(rows):
            if idx >= N:
                break
            params[3*idx] = xs[i] + rng.uniform(-0.02, 0.02)*width
            params[3*idx+1] = ys[j] + rng.uniform(-0.02, 0.02)*height
            params[3*idx+2] = 0.03 * min(width, height)  # small initial radius
            idx += 1
    return params

def _try_aspect(aspect, seed=42, max_iter=5000):
    """Try to optimize for a given aspect ratio. width/height = aspect, width+height=2."""
    if _time_left() < 5.0:
        return None
    width = 2.0 * aspect / (1.0 + aspect)
    height = 2.0 / (1.0 + aspect)
    
    x0 = _make_initial(width, height, seed)
    
    # Bounds: x in [0, width], y in [0, height], r in [0, min(width,height)/2]
    rmax = min(width, height) / 2.0
    bounds = []
    for i in range(N):
        bounds.append((0.0, width))
        bounds.append((0.0, height))
        bounds.append((0.0, rmax))
    
    result = minimize(
        _penalty_objective, x0, args=(width, height),
        method='L-BFGS-B', bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    if _time_left() < 5.0:
        return None
    
    if not result.success and not hasattr(result, 'x'):
        return None
    
    params = result.x
    circles = np.array([[params[3*i], params[3*i+1], params[3*i+2]] for i in range(N)])
    
    # Check validity
    if _is_valid(circles, width, height, tol=1e-5):
        return circles, width, height
    else:
        # Try to repair: clamp to boundaries and reduce radii slightly
        for i in range(N):
            x, y, r = circles[i]
            r = max(0, min(r, x, width-x, y, height-y))
            circles[i] = [x, y, r]
        if _is_valid(circles, width, height, tol=1e-4):
            return circles, width, height
        return None

def construct():
    global _START
    _START = time.time()
    
    best_circles = None
    best_width = None
    best_height = None
    best_sum = -np.inf
    
    # Try several aspect ratios
    aspects = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 2.0, 0.4, 2.5, 0.33, 3.0]
    
    for aspect in aspects:
        if _time_left() < 10.0:
            break
        result = _try_aspect(aspect, seed=42, max_iter=3000)
        if result is None:
            continue
        circles, w, h = result
        s = np.sum(circles[:, 2])
        if s > best_sum:
            best_sum = s
            best_circles = circles.copy()
            best_width = w
            best_height = h
    
    if best_circles is None:
        # Fallback: simple grid
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                cells.append([x, y, r])
        # Scale to perimeter 4: width=height=1
        return np.array(cells)
    
    # Final validation
    if not _is_valid(best_circles, best_width, best_height, tol=1e-4):
        # Last resort fallback
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                cells.append([x, y, r])
        return np.array(cells)
    
    return best_circles
