import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 130.0  # safety margin before 150s

def _compute_radii(x, y, W, H):
    """Compute max radius for each circle given centers and box."""
    n = len(x)
    r = np.empty(n)
    # Distance to walls
    r = np.minimum(x, W - x)
    r = np.minimum(r, y)
    r = np.minimum(r, H - y)
    # Distance to other circles
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            # Each circle can be at most d/2 from the other
            half = 0.5 * d
            r[i] = min(r[i], half)
            r[j] = min(r[j], half)
    return r

def _sum_radii(x, y, W, H):
    r = _compute_radii(x, y, W, H)
    return np.sum(r), r

def _validate(x, y, r, W, H, tol=1e-7):
    """Check if circles are valid (no overlap, inside box)."""
    if np.any(r <= 0):
        return False
    if np.any(x < r - tol) or np.any(x > W - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > H - tol):
        return False
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if d < r[i] + r[j] - tol:
                return False
    return True

def _objective(v, W, H):
    """Objective: maximize sum of Voronoi radii, penalize overlaps."""
    n = N
    x = v[:n]
    y = v[n:2*n]
    
    # Compute Voronoi radii
    r = _compute_radii(x, y, W, H)
    val = np.sum(r)
    
    # Penalty for any circle being too close to wall (shouldn't happen with Voronoi)
    # and for numerical issues
    pen = 0.0
    # Soft penalty: if a circle's center is within r of wall, it's fine (that's the boundary)
    # The Voronoi computation already handles this.
    
    return -val

def _optimize_for_aspect(W, H, start_x, start_y, max_iter=300):
    """Optimize positions for given aspect ratio."""
    n = N
    x0 = np.concatenate([start_x, start_y])
    
    bounds = [(0.0, W)] * n + [(0.0, H)] * n
    
    res = minimize(_objective, x0, args=(W, H), method='L-BFGS-B',
                   bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
    
    x = res.x[:n]
    y = res.x[n:2*n]
    r = _compute_radii(x, y, W, H)
    
    if _validate(x, y, r, W, H):
        return x, y, r, np.sum(r)
    return None

def _initial_grid(W, H, cols, rows, jitter=0.0):
    """Generate grid initialization with optional jitter."""
    n = N
    xs = np.zeros(n)
    ys = np.zeros(n)
    k = 0
    for i in range(cols):
        for j in range(rows):
            if k >= n:
                break
            cx = (i + 0.5) * W / cols
            cy = (j + 0.5) * H / rows
            if jitter > 0:
                cx += (np.random.rand() - 0.5) * W / cols * jitter
                cy += (np.random.rand() - 0.5) * H / rows * jitter
            xs[k] = np.clip(cx, 0.01, W - 0.01)
            ys[k] = np.clip(cy, 0.01, H - 0.01)
            k += 1
    return xs, ys

def _initial_random(W, H, seed=None):
    """Random initialization."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    xs = rng.uniform(0.05, W - 0.05, N)
    ys = rng.uniform(0.05, H - 0.05, N)
    return xs, ys

def construct():
    t0 = time.time()
    best_val = -1
    best_result = None
    
    # Aspect ratios to try (W + H = 2)
    # Due to symmetry, W in [1, 2) covers all unique cases
    aspect_ratios = [
        1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5,
        0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55
    ]
    
    # Grid configurations: (cols, rows)
    grid_configs = [
        (4, 6), (5, 5), (6, 4), (3, 7), (7, 3), (4, 5), (5, 4),
        (6, 4), (4, 6), (5, 5)
    ]
    
    restarts_per_aspect = 3
    max_iter_per_restart = 200
    
    total_attempts = len(aspect_ratios) * restarts_per_aspect
    
    for ai, W in enumerate(aspect_ratios):
        H = 2.0 - W
        if H <= 0.1:
            continue
        
        for ri in range(restarts_per_aspect):
            # Check deadline
            elapsed = time.time() - t0
            if elapsed > DEADLINE - 5.0:
                break
            
            # Choose initialization
            if ri == 0:
                # Grid init
                cols, rows = grid_configs[ai % len(grid_configs)]
                start_x, start_y = _initial_grid(W, H, cols, rows)
            elif ri == 1:
                # Slightly jittered grid
                cols, rows = grid_configs[(ai + 1) % len(grid_configs)]
                start_x, start_y = _initial_grid(W, H, cols, rows, jitter=0.3)
            else:
                # Random init
                start_x, start_y = _initial_random(W, H, seed=ai * 100 + ri)
            
            result = _optimize_for_aspect(W, H, start_x, start_y, max_iter=max_iter_per_restart)
            
            if result is not None:
                x, y, r, val = result
                if val > best_val:
                    best_val = val
                    best_result = np.column_stack((x, y, r))
        
        # Check deadline after each aspect ratio
        if time.time() - t0 > DEADLINE - 5.0:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: simple grid in 1x1 square
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append([cx, cy, r])
    return np.array(cells[:N])
