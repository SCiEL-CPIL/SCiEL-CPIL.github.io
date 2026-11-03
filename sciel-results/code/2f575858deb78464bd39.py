"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

# EVOLVE-BLOCK-START

def _violation_and_radius(x, W, H):
    """
    x: shape (N, 3) with columns [cx, cy, r]
    Returns (violation, sum_r)
    violation = sum of max(0, r_i + r_j - dist_ij) for all pairs
              + sum of max(0, r_i - min(cx_i, W-cx_i, cy_i, H-cy_i)) for walls
    """
    n = x.shape[0]
    cx = x[:, 0]
    cy = x[:, 1]
    r = x[:, 2]
    
    # Wall violations
    # Distance to each wall
    d_left = cx
    d_right = W - cx
    d_bottom = cy
    d_top = H - cy
    d_min = np.minimum.reduce([d_left, d_right, d_bottom, d_top])
    wall_viol = np.maximum(0, r - d_min)
    
    # Pairwise violations
    # Use broadcasting for all pairs
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    r_sum = r[:, None] + r[None, :]
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    pair_viol = np.maximum(0, r_sum[iu] - dist[iu])
    
    total_viol = np.sum(wall_viol) + np.sum(pair_viol)
    sum_r = np.sum(r)
    return total_viol, sum_r


def _objective(x, W, H, lam):
    """
    Objective to minimize: lam * violation - sum_r
    We want violation -> 0 and sum_r -> max
    """
    viol, sr = _violation_and_radius(x, W, H)
    return lam * viol - sr


def _penalty_objective(x, W, H, mu):
    """
    Penalty method: minimize -sum_r + mu * violation^2
    """
    viol, sr = _violation_and_radius(x, W, H)
    return -sr + mu * viol * viol


def _greedy_init(W, H, n_circles=N):
    """
    Greedy placement: place circles one by one at the position that maximizes
    the radius while maintaining non-overlap with previously placed circles
    and staying within bounds.
    """
    placed = []  # list of (cx, cy, r)
    
    # Candidate positions on a grid
    nx, ny = 15, 15
    xs = np.linspace(0.05 * W, 0.95 * W, nx)
    ys = np.linspace(0.05 * H, 0.95 * H, ny)
    
    for _ in range(n_circles):
        best_pos = None
        best_r = 0
        for cx in xs:
            for cy in ys:
                # Max radius allowed by walls
                r_wall = min(cx, W - cx, cy, H - cy)
                if r_wall < 1e-6:
                    continue
                # Check against placed circles
                r_max = r_wall
                for (px, py, pr) in placed:
                    d = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                    r_max = min(r_max, d - pr)
                if r_max > best_r:
                    best_r = r_max
                    best_pos = (cx, cy)
        
        if best_pos is not None and best_r > 1e-6:
            placed.append((best_pos[0], best_pos[1], best_r))
    
    # Pad with tiny circles if needed
    while len(placed) < n_circles:
        placed.append((W / 2, H / 2, 1e-8))
    
    return np.array(placed[:n_circles])


def _grid_init(W, H, n_circles=N):
    """
    Grid-based initialization with slight perturbation.
    """
    # Find grid dimensions
    best = None
    for cols in range(1, n_circles + 1):
        rows = int(np.ceil(n_circles / cols))
        r = min(W / (2 * cols), H / (2 * rows))
        if best is None or r > best[0]:
            best = (r, cols, rows)
    
    _, cols, rows = best
    cells = []
    for i in range(cols):
        for j in range(rows):
            if len(cells) >= n_circles:
                break
            cx = (i + 0.5) * W / cols
            cy = (j + 0.5) * H / rows
            r = min(W / (2 * cols), H / (2 * rows)) * 0.95
            cells.append([cx, cy, r])
    
    # Pad
    while len(cells) < n_circles:
        cells.append([W / 2, H / 2, 1e-8])
    
    return np.array(cells[:n_circles])


def _optimize_aspect(W, H, x0, time_limit=30):
    """
    Optimize circle positions and radii given fixed W, H.
    Uses L-BFGS-B with penalty method.
    """
    n = x0.shape[0]
    x_flat = x0.flatten()
    
    # Bounds: cx in [0, W], cy in [0, H], r in [0, min(W,H)/2]
    r_max = min(W, H) / 2
    bounds = []
    for i in range(n):
        bounds.append((1e-8, W - 1e-8))  # cx
        bounds.append((1e-8, H - 1e-8))  # cy
        bounds.append((1e-8, r_max))      # r
    
    start_time = time.time()
    best_x = x_flat.copy()
    best_obj = _penalty_objective(x0, W, H, 1.0)
    
    # Adaptive penalty
    mu = 1.0
    for iteration in range(20):
        if time.time() - start_time > time_limit * 0.8:
            break
        
        def obj_func(xf):
            x = xf.reshape(n, 3)
            return _penalty_objective(x, W, H, mu)
        
        try:
            res = minimize(obj_func, best_x, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success or res.fun < best_obj:
                best_x = res.x
                best_obj = res.fun
        except Exception:
            pass
        
        # Check violation
        x_curr = best_x.reshape(n, 3)
        viol, sr = _violation_and_radius(x_curr, W, H)
        
        if viol < 1e-6:
            # Feasible, try to increase radii
            # Scale up radii slightly and re-optimize
            x_curr[:, 2] *= 1.01
            best_x = x_curr.flatten()
            mu = 10.0
        else:
            mu *= 2.0
            if mu > 1e6:
                break
    
    return best_x.reshape(n, 3)


def construct():
    deadline = time.time() + 130  # Safety margin from 150s limit
    
    best_result = None
    best_value = -np.inf
    
    # Scan aspect ratios
    # Perimeter 4: 2(W+H) = 4 => W+H = 2
    # W in (0, 2), H = 2 - W
    # Try a range of aspect ratios
    W_vals = np.linspace(0.5, 1.5, 11)  # W from 0.5 to 1.5, H from 1.5 to 0.5
    
    for W in W_vals:
        if time.time() > deadline:
            break
        H = 2.0 - W
        
        # Try multiple initializations
        for init_method in ['greedy', 'grid']:
            if time.time() > deadline:
                break
            
            if init_method == 'greedy':
                x0 = _greedy_init(W, H)
            else:
                x0 = _grid_init(W, H)
            
            # Optimize
            x_opt = _optimize_aspect(W, H, x0, time_limit=20)
            
            # Check validity
            viol, sr = _violation_and_radius(x_opt, W, H)
            if viol < 1e-4:
                if sr > best_value:
                    best_value = sr
                    best_result = x_opt.copy()
    
    # If no valid result found, fall back to grid
    if best_result is None:
        W = 1.0
        H = 1.0
        x0 = _grid_init(W, H)
        x_opt = _optimize_aspect(W, H, x0, time_limit=20)
        viol, sr = _violation_and_radius(x_opt, W, H)
        if viol < 1e-3:
            best_result = x_opt
        else:
            # Last resort: very small circles
            best_result = np.array([[W/2, H/2, 0.01]] * N)
    
    return best_result

# EVOLVE-BLOCK-END
