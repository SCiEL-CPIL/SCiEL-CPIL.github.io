import numpy as np
from scipy.optimize import minimize
import time

N = 21
_DEADLINE = 135.0  # safety margin before 150s kill
SOFTMIN_TAU = 0.01  # temperature for soft-min smoothing

def _compute_radii(x, y, W, H):
    """Compute max radius for each circle given centers and box (Voronoi-style)."""
    n = len(x)
    # Distance to walls
    r = np.minimum(x, W - x)
    r = np.minimum(r, y)
    r = np.minimum(r, H - y)
    # Distance to other circles
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            half = 0.5 * d
            if half < r[i]:
                r[i] = half
            if half < r[j]:
                r[j] = half
    return r

def _softmin_radii(x, y, W, H, tau=SOFTMIN_TAU):
    """Compute smoothed radii using soft-min for differentiability."""
    n = len(x)
    r = np.zeros(n)
    for i in range(n):
        # Wall distances
        d_left = x[i]
        d_right = W - x[i]
        d_bottom = y[i]
        d_top = H - y[i]
        
        # Start with wall constraints
        vals = [d_left, d_right, d_bottom, d_top]
        
        # Add distances to other circles
        for j in range(n):
            if j == i:
                continue
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            vals.append(0.5 * d)
        
        # Soft-min of all values
        vals = np.array(vals)
        # Clip to avoid overflow
        vals_clipped = np.clip(vals, 1e-8, None)
        log_sum = np.log(np.sum(np.exp(-vals_clipped / tau)))
        r[i] = -tau * log_sum
    
    return r

def _validate(x, y, r, W, H, tol=1e-6):
    """Check if circles are valid (no overlap, inside box)."""
    if np.any(r <= 0):
        return False
    if np.any(x < r - tol) or np.any(x > W - r + tol):
        return False
    if np.any(y < r - tol) or np.any(y > H - r + tol):
        return False
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if d < r[i] + r[j] - tol:
                return False
    return True

def _objective(v, W, H):
    """Objective: maximize sum of Voronoi radii."""
    n = N
    x = v[:n]
    y = v[n:2*n]
    r = _compute_radii(x, y, W, H)
    return -np.sum(r)

def _objective_soft(v, W, H, tau=SOFTMIN_TAU):
    """Objective using soft-min radii for smoother optimization."""
    n = N
    x = v[:n]
    y = v[n:2*n]
    r = _softmin_radii(x, y, W, H, tau)
    return -np.sum(r)

def _initial_grid(W, H, cols, rows, jitter=0.0, seed=None):
    """Generate grid initialization with optional jitter."""
    n = N
    xs = np.zeros(n)
    ys = np.zeros(n)
    k = 0
    rng = np.random.RandomState(seed) if seed is not None else np.random
    for i in range(cols):
        for j in range(rows):
            if k >= n:
                break
            cx = (i + 0.5) * W / cols
            cy = (j + 0.5) * H / rows
            if jitter > 0:
                cx += (rng.rand() - 0.5) * W / cols * jitter
                cy += (rng.rand() - 0.5) * H / rows * jitter
            xs[k] = np.clip(cx, 0.01, W - 0.01)
            ys[k] = np.clip(cy, 0.01, H - 0.01)
            k += 1
    return xs, ys

def _initial_random(W, H, seed=None):
    """Random initialization."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    xs = rng.uniform(0.05, W - 0.05, N)
    ys = rng.uniform(0.05, H - 0.05, N)
    return xs, ys

def _optimize_for_aspect(W, H, start_x, start_y, start_time, time_budget, max_iter=500):
    """Optimize positions for given aspect ratio."""
    n = N
    x0 = np.concatenate([start_x, start_y])
    bounds = [(0.0, W)] * n + [(0.0, H)] * n
    
    try:
        # First pass: use soft-min objective for smoother landscape
        res = minimize(_objective_soft, x0, args=(W, H), method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
        
        x = res.x[:n]
        y = res.x[n:2*n]
        
        # Check time
        if time.time() - start_time > time_budget:
            r = _compute_radii(x, y, W, H)
            if _validate(x, y, r, W, H):
                return x, y, r, np.sum(r)
            return None
        
        # Second pass: refine with exact (nonsmooth) objective
        res2 = minimize(_objective, res.x, args=(W, H), method='L-BFGS-B',
                        bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
        
        x = res2.x[:n]
        y = res2.x[n:2*n]
    except Exception:
        return None
    
    r = _compute_radii(x, y, W, H)
    
    if _validate(x, y, r, W, H):
        return x, y, r, np.sum(r)
    return None

def _local_refine(x, y, W, H, start_time, time_budget):
    """Further refine using Nelder-Mead on a subset of coordinates."""
    n = N
    x0 = np.concatenate([x, y])
    
    # Try Nelder-Mead for local refinement
    try:
        res = minimize(_objective, x0, args=(W, H), method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
    except Exception:
        pass
    else:
        x_new = res.x[:n]
        y_new = res.x[n:2*n]
        # Clip to bounds
        x_new = np.clip(x_new, 0.0, W)
        y_new = np.clip(y_new, 0.0, H)
        r_new = _compute_radii(x_new, y_new, W, H)
        if _validate(x_new, y_new, r_new, W, H):
            val_new = np.sum(r_new)
            r_old = _compute_radii(x, y, W, H)
            val_old = np.sum(r_old)
            if val_new > val_old:
                # Final high-precision polish with tighter tolerances
                try:
                    res3 = minimize(_objective, res.x, args=(W, H), method='L-BFGS-B',
                                   bounds=[(0.0, W)] * n + [(0.0, H)] * n,
                                   options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
                    x_pol = np.clip(res3.x[:n], 0.0, W)
                    y_pol = np.clip(res3.x[n:2*n], 0.0, H)
                    r_pol = _compute_radii(x_pol, y_pol, W, H)
                    if _validate(x_pol, y_pol, r_pol, W, H):
                        val_pol = np.sum(r_pol)
                        if val_pol > val_new:
                            return x_pol, y_pol, r_pol, val_pol
                except Exception:
                    pass
                return x_new, y_new, r_new, val_new
    
    r = _compute_radii(x, y, W, H)
    return x, y, r, np.sum(r)

def construct():
    start_time = time.time()
    time_budget = _DEADLINE - start_time - 5.0  # 5s safety margin
    
    best_val = -1
    best_res = None
    
    # Aspect ratios to try (W + H = 2)
    # Due to symmetry, W in [1, 2) covers all unique cases, but we try both
    aspect_ratios = [
        1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.5,
        0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5
    ]
    
    # Grid configurations: (cols, rows)
    grid_configs = [
        (4, 6), (5, 5), (6, 4), (3, 7), (7, 3), (4, 5), (5, 4),
        (6, 4), (4, 6), (5, 5), (3, 7), (7, 3)
    ]
    
    restarts_per_aspect = 4
    max_iter_per_restart = 500
    
    for ai, W in enumerate(aspect_ratios):
        H = 2.0 - W
        if H <= 0.1:
            continue
        
        for ri in range(restarts_per_aspect):
            # Check deadline
            elapsed = time.time() - start_time
            if elapsed > time_budget:
                break
            
            # Choose initialization
            if ri == 0:
                # Grid init
                cols, rows = grid_configs[ai % len(grid_configs)]
                start_x, start_y = _initial_grid(W, H, cols, rows, seed=ai)
            elif ri == 1:
                # Slightly jittered grid
                cols, rows = grid_configs[(ai + 1) % len(grid_configs)]
                start_x, start_y = _initial_grid(W, H, cols, rows, jitter=0.3, seed=ai + 100)
            elif ri == 2:
                # Random init
                start_x, start_y = _initial_random(W, H, seed=ai + 200)
            else:
                # Different grid with more jitter
                cols, rows = grid_configs[(ai + 2) % len(grid_configs)]
                start_x, start_y = _initial_grid(W, H, cols, rows, jitter=0.5, seed=ai + 300)
            
            result = _optimize_for_aspect(W, H, start_x, start_y, start_time, time_budget, max_iter_per_restart)
            
            if result is not None:
                x, y, r, val = result
                if val > best_val:
                    # Try local refinement
                    x, y, r, val = _local_refine(x, y, W, H, start_time, time_budget)
                    if val > best_val:
                        best_val = val
                        best_res = np.column_stack((x, y, r))
    
    if best_res is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_res
