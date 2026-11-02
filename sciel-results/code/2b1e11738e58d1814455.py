"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 circles."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
DEADLINE = 115.0  # seconds wall-clock limit (safety margin for 120s kill)


def _check_validity(arr, tol=1e-9):
    """Check if circles are valid: disjoint interiors, inside bounding box, perimeter <= 4."""
    n = arr.shape[0]
    xs, ys, rs = arr[:, 0], arr[:, 1], arr[:, 2]
    
    # All radii must be positive
    if np.any(rs <= 0):
        return False
    
    # Bounding box: the minimal axis-aligned rectangle containing all circles
    x_ext_min = np.min(xs - rs)
    x_ext_max = np.max(xs + rs)
    y_ext_min = np.min(ys - rs)
    y_ext_max = np.max(ys + rs)
    
    perim = 2.0 * ((x_ext_max - x_ext_min) + (y_ext_max - y_ext_min))
    if perim > 4.0 + tol:
        return False
    
    # Disjoint interiors: distance between centers >= sum of radii
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    
    return True


def _is_valid_in_rect(arr, w, h, tol=1e-9):
    """Fast validity check for circles in a fixed rectangle [0,w] x [0,h]."""
    xs, ys, rs = arr[:, 0], arr[:, 1], arr[:, 2]
    
    if np.any(rs <= 0):
        return False
    
    # Boundary check
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    
    # Pairwise check
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    
    return True


def _max_radius_for_circle(arr, i, w, h, tol=1e-10):
    """Compute the maximum radius for circle i given other circles and boundaries."""
    x, y = arr[i, 0], arr[i, 1]
    
    # Boundary constraints
    r_max = min(x, w - x, y, h - y)
    
    # Pairwise constraints
    for j in range(N):
        if j == i:
            continue
        dx = x - arr[j, 0]
        dy = y - arr[j, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        # Need: dist >= r + r_j => r <= dist - r_j
        r_max = min(r_max, dist - arr[j, 2])
    
    return max(r_max, 0.0)


def _expand_circles(arr, w, h, max_iters=500, tol=1e-12):
    """Iteratively expand each circle to its maximum possible radius."""
    for _ in range(max_iters):
        improved = False
        for i in range(N):
            new_r = _max_radius_for_circle(arr, i, w, h, tol)
            if new_r > arr[i, 2] + tol:
                arr[i, 2] = new_r
                improved = True
        if not improved:
            break
    return arr


def _hexagonal_init(w, h):
    """Create a hexagonal close-packed initial configuration."""
    arr = np.zeros((N, 3))
    
    # Try different row offsets to fill 21 circles nicely
    # For a hexagonal packing, rows alternate offset
    # Let's try to fit as many as possible in a hex pattern
    
    # Estimate radius from area
    area = w * h
    # 21 circles, each with area ~ pi*r^2, packing efficiency ~ 0.9 for hex
    # 21 * pi * r^2 * 0.9 = area => r = sqrt(area / (21*pi*0.9))
    r_est = np.sqrt(area / (N * np.pi * 0.85))
    r_est = min(r_est, w / 5.0, h / 5.0)  # Cap to ensure fit
    
    # Try different number of columns
    best_arr = None
    best_sum = 0.0
    
    for n_cols in range(4, 8):
        n_rows = int(np.ceil(N / n_cols))
        
        # Compute spacing for hexagonal pattern
        # Horizontal spacing: 2*r + gap
        # Vertical spacing: sqrt(3)*r + gap
        # For hex packing, row offset by r
        
        # Let's just place them in a grid with hex offset for odd rows
        spacing_x = w / (n_cols + 0.5)  # Extra margin
        spacing_y = h / (n_rows + 0.5)
        
        r = min(spacing_x, spacing_y) * 0.45
        
        count = 0
        valid = True
        for row in range(n_rows):
            offset = 0.0 if row % 2 == 0 else spacing_x / 2.0
            for col in range(n_cols):
                if count >= N:
                    break
                x = (col + 0.75 + offset / spacing_x * 0.5) * spacing_x
                y = (row + 0.75) * spacing_y
                
                # Keep within bounds
                x = min(max(x, r), w - r)
                y = min(max(y, r), h - r)
                
                arr[count, 0] = x
                arr[count, 1] = y
                arr[count, 2] = r
                count += 1
        
        if count >= N:
            arr = arr[:N]
            if _is_valid_in_rect(arr, w, h):
                arr_exp = _expand_circles(arr.copy(), w, h)
                s = np.sum(arr_exp[:, 2])
                if s > best_sum:
                    best_sum = s
                    best_arr = arr_exp.copy()
    
    if best_arr is not None:
        return best_arr
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            arr[count, 0] = (i + 0.5) * w / side
            arr[count, 1] = (j + 0.5) * h / side
            arr[count, 2] = min(w, h) / (2 * side) * 0.8
            count += 1
    arr = arr[:N]
    
    if not _is_valid_in_rect(arr, w, h):
        arr[:, 2] *= 0.7
    return _expand_circles(arr, w, h)


def _slsqp_pack(w, h, arr_init, time_limit=8.0):
    """
    Use SLSQP to maximize sum of radii subject to non-overlap and boundary constraints.
    """
    start = time.time()
    
    # Precompute pairs
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((i, j))
    n_pairs = len(pairs)
    
    def objective(z):
        r = z[2::3]
        return -np.sum(r)
    
    def objective_grad(z):
        grad = np.zeros(3 * N)
        grad[2::3] = -1.0
        return grad
    
    def constraints(z):
        if time.time() - start > time_limit:
            return []
        
        cons = []
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        # Boundary constraints (vectorized)
        bnd = np.zeros(4 * N)
        bnd[0::4] = x - r
        bnd[1::4] = w - x - r
        bnd[2::4] = y - r
        bnd[3::4] = h - y - r
        
        def bnd_fun(z):
            x = z[0::3]
            y = z[1::3]
            r = z[2::3]
            b = np.zeros(4 * N)
            b[0::4] = x - r
            b[1::4] = w - x - r
            b[2::4] = y - r
            b[3::4] = h - y - r
            return b
        
        cons.append({
            'type': 'ineq',
            'fun': bnd_fun
        })
        
        # Pairwise constraints (vectorized)
        def pairwise_fun(z):
            x = z[0::3]
            y = z[1::3]
            r = z[2::3]
            vals = np.zeros(n_pairs)
            for k, (i, j) in enumerate(pairs):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                sum_r = r[i] + r[j]
                vals[k] = dist_sq - sum_r * sum_r
            return vals
        
        cons.append({
            'type': 'ineq',
            'fun': pairwise_fun
        })
        
        return cons
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((0, min(w, h) / 2.0))
    
    # Initial guess
    z0 = np.zeros(3 * N)
    for i in range(N):
        z0[3 * i] = arr_init[i, 0]
        z0[3 * i + 1] = arr_init[i, 1]
        z0[3 * i + 2] = arr_init[i, 2]
    
    try:
        res = minimize(objective, z0, method='SLSQP', bounds=bounds,
                       jac=objective_grad,
                       constraints=constraints(z0),
                       options={'maxiter': 300, 'ftol': 1e-10})
        
        if res.success and time.time() - start < time_limit:
            arr_res = np.zeros((N, 3))
            arr_res[:, 0] = res.x[0::3]
            arr_res[:, 1] = res.x[1::3]
            arr_res[:, 2] = res.x[2::3]
            
            # Ensure validity and expand
            if _is_valid_in_rect(arr_res, w, h):
                arr_res = _expand_circles(arr_res, w, h)
            return arr_res
    except Exception:
        pass
        
    return None


def _random_init(w, h, seed=None):
    """Create random initial packing."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    arr = np.zeros((N, 3))
    for i in range(N):
        arr[i, 0] = random.uniform(0.1 * w, 0.9 * w)
        arr[i, 1] = random.uniform(0.1 * h, 0.9 * h)
        arr[i, 2] = random.uniform(0.03, 0.12) * min(w, h)
    
    return arr


def _jiggle_polish(arr, w, h, n_iters=200, step=0.01, time_limit=5.0):
    """Simple jiggle-based local search to improve packing."""
    start = time.time()
    best = arr.copy()
    best_sum = np.sum(arr[:, 2])
    
    for it in range(n_iters):
        if time.time() - start > time_limit:
            break
        
        # Try moving a random circle slightly
        i = random.randint(0, N - 1)
        dx = random.uniform(-step, step)
        dy = random.uniform(-step, step)
        
        arr[i, 0] += dx
        arr[i, 1] += dy
        
        # Clamp to bounds
        arr[i, 0] = min(max(arr[i, 0], arr[i, 2]), w - arr[i, 2])
        arr[i, 1] = min(max(arr[i, 1], arr[i, 2]), h - arr[i, 2])
        
        # Expand radii
        arr = _expand_circles(arr, w, h, max_iters=50)
        s = np.sum(arr[:, 2])
        
        if s > best_sum:
            best_sum = s
            best = arr.copy()
        else:
            # Revert
            arr[i, 0] -= dx
            arr[i, 1] -= dy
            arr[i, 0] = min(max(arr[i, 0], arr[i, 2]), w - arr[i, 2])
            arr[i, 1] = min(max(arr[i, 1], arr[i, 2]), h - arr[i, 2])
    
    return best


def construct():
    start_time = time.time()
    
    best_arr = None
    best_sum = 0.0
    
    # Try different aspect ratios
    aspects = [1.0, 1.2, 0.8, 1.5, 0.67, 1.33, 0.75, 1.1, 0.9, 2.0, 0.5]
    
    for aspect in aspects:
        if time.time() - start_time > DEADLINE - 10.0:
            break
        
        h = 2.0 / (aspect + 1.0)
        w = aspect * h
        
        # Try hexagonal init
        arr_hex = _hexagonal_init(w, h)
        if arr_hex is not None:
            s = np.sum(arr_hex[:, 2])
            if s > best_sum:
                best_sum = s
                best_arr = arr_hex.copy()
            
            # Try SLSQP refinement
            arr_opt = _slsqp_pack(w, h, arr_hex, time_limit=6.0)
            if arr_opt is not None:
                s = np.sum(arr_opt[:, 2])
                if s > best_sum:
                    best_sum = s
                    best_arr = arr_opt.copy()
            
            # Jiggle polish
            if time.time() - start_time < DEADLINE - 5.0:
                arr_jig = _jiggle_polish(best_arr.copy(), w, h, n_iters=100, step=0.02, time_limit=3.0)
                s = np.sum(arr_jig[:, 2])
                if s > best_sum:
                    best_sum = s
                    best_arr = arr_jig.copy()
        
        # Try random inits
        for seed in [42, 123, 456]:
            if time.time() - start_time > DEADLINE - 5.0:
                break
            
            arr_rand = _random_init(w, h, seed=seed)
            if not _is_valid_in_rect(arr_rand, w, h):
                arr_rand[:, 2] *= 0.8
                if not _is_valid_in_rect(arr_rand, w, h):
                    continue
            
            arr_exp = _expand_circles(arr_rand, w, h)
            s = np.sum(arr_exp[:, 2])
            if s > best_sum:
                best_sum = s
                best_arr = arr_exp.copy()
            
            # Try SLSQP on random init
            arr_opt = _slsqp_pack(w, h, arr_exp, time_limit=5.0)
            if arr_opt is not None:
                s = np.sum(arr_opt[:, 2])
                if s > best_sum:
                    best_sum = s
                    best_arr = arr_opt.copy()
    
    # Final fallback
    if best_arr is None or not _check_validity(best_arr):
        w = h = 1.0
        side = 5
        best_arr = np.zeros((N, 3))
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                best_arr[count, 0] = (i + 0.5) / side
                best_arr[count, 1] = (j + 0.5) / side
                best_arr[count, 2] = 0.5 / side - 0.01
                count += 1
    
    return best_arr
