"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 circles."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
DEADLINE = 120.0  # seconds wall-clock limit


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


def _max_radius_for_circle(arr, i, w, h, tol=1e-10):
    """Compute the maximum radius for circle i given other circles and boundaries."""
    x, y, r = arr[i, 0], arr[i, 1], arr[i, 2]
    
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


def _is_valid(arr, w, h, tol=1e-9):
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


def _expand_circles(arr, w, h, max_iters=1000, tol=1e-12):
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


def _polish_result(arr, w, h, time_limit=5.0):
    """Polish the packing by moving circles to free up space and expanding."""
    start = time.time()
    best_sum = np.sum(arr[:, 2])
    arr_best = arr.copy()
    
    n_steps = 200
    step_size = 0.02
    
    for step in range(n_steps):
        if time.time() - start > time_limit:
            break
        
        # Try moving a random circle
        i = random.randint(0, N - 1)
        dx = random.uniform(-step_size, step_size)
        dy = random.uniform(-step_size, step_size)
        
        arr_new = arr.copy()
        arr_new[i, 0] = np.clip(arr_new[i, 0] + dx, 0, w)
        arr_new[i, 1] = np.clip(arr_new[i, 1] + dy, 0, h)
        
        if _is_valid(arr_new, w, h):
            # Try to expand all circles
            arr_expanded = _expand_circles(arr_new, w, h, max_iters=100)
            new_sum = np.sum(arr_expanded[:, 2])
            if new_sum > best_sum + 1e-10:
                arr = arr_expanded
                arr_best = arr.copy()
                best_sum = new_sum
                step_size *= 0.95  # gradually reduce step size
    
    return arr_best, best_sum


def _constraint_refine(arr, w, h, time_limit=10.0):
    """Refine positions to maximize minimum clearance, then expand radii."""
    start = time.time()
    
    # Try to optimize positions to maximize the minimum gap
    def objective(z):
        xs = z[0::2]
        ys = z[1::2]
        min_gap = float('inf')
        for i in range(N):
            # Boundary gaps
            min_gap = min(min_gap, xs[i], w - xs[i], ys[i], h - ys[i])
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                min_gap = min(min_gap, dist - arr[i, 2] - arr[j, 2])
        return -min_gap
    
    # Initial guess
    z0 = np.zeros(2 * N)
    for i in range(N):
        z0[2*i] = arr[i, 0]
        z0[2*i + 1] = arr[i, 1]
    
    bounds = []
    for i in range(N):
        bounds.append((0, w))
        bounds.append((0, h))
    
    try:
        result = minimize(objective, z0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 50, 'ftol': 1e-10})
        if time.time() - start < time_limit:
            arr_new = arr.copy()
            arr_new[:, 0] = result.x[0::2]
            arr_new[:, 1] = result.x[1::2]
            if _is_valid(arr_new, w, h):
                arr_expanded = _expand_circles(arr_new, w, h)
                return arr_expanded
    except Exception:
        pass
    
    return arr


def _force_relax_unified(arr, w, h, time_limit=15.0):
    """Relax constraints by shrinking radii slightly, then re-expand."""
    start = time.time()
    
    # Slightly shrink all radii
    arr_shrunk = arr.copy()
    arr_shrunk[:, 2] *= 0.95
    
    # Run polish
    arr_polished, s = _polish_result(arr_shrunk, w, h, time_limit=time_limit * 0.5)
    
    # Run constraint refine
    if time.time() - start < time_limit * 0.7:
        arr_refined = _constraint_refine(arr_polished, w, h, time_limit=time_limit * 0.3)
        if np.sum(arr_refined[:, 2]) > s:
            arr_polished = arr_refined
    
    return arr_polished


def _expansion_relaxation_polish(arr, w, h, time_limit=20.0):
    """Combined expansion and relaxation polish."""
    start = time.time()
    best = arr.copy()
    best_sum = np.sum(arr[:, 2])
    
    # First, fully expand
    arr_expanded = _expand_circles(arr, w, h)
    s = np.sum(arr_expanded[:, 2])
    if s > best_sum:
        best = arr_expanded
        best_sum = s
    
    # Then polish
    if time.time() - start < time_limit * 0.5:
        arr_polished, s2 = _polish_result(best, w, h, time_limit=time_limit * 0.4)
        if s2 > best_sum:
            best = arr_polished
            best_sum = s2
    
    # Then relax and re-expand
    if time.time() - start < time_limit * 0.7:
        arr_relaxed = _force_relax_unified(best, w, h, time_limit=time_limit * 0.3)
        s3 = np.sum(arr_relaxed[:, 2])
        if s3 > best_sum:
            best = arr_relaxed
            best_sum = s3
    
    return best


def _initial_hexagonal(w, h):
    """Create initial hexagonal packing."""
    arr = np.zeros((N, 3))
    
    # Try to fit hexagonal pattern
    # For a hexagonal packing with radius r, the horizontal spacing is 2*r*sqrt(3)/2 = r*sqrt(3)
    # and vertical spacing is r*1.5
    
    # Estimate radius
    r_est = min(w, h) / (2 * 5) * 0.9
    
    row = 0
    col = 0
    idx = 0
    while idx < N:
        x = (col + 0.5) * r_est * np.sqrt(3) + (r_est * np.sqrt(3) / 2 if row % 2 else 0)
        y = (row + 0.5) * r_est * 1.5 + r_est
        
        if x + r_est <= w and y + r_est <= h:
            arr[idx, 0] = x
            arr[idx, 1] = y
            arr[idx, 2] = r_est
            idx += 1
            col += 1
        else:
            col = 0
            row += 1
            if row > 10:  # safety
                break
    
    # Fill remaining with grid
    if idx < N:
        side = 5
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                arr[idx, 0] = (i + 0.5) * w / side
                arr[idx, 1] = (j + 0.5) * h / side
                arr[idx, 2] = min(w, h) / (2 * side) * 0.8
                idx += 1
    
    return arr


def _initial_grid(w, h):
    """Create initial grid packing."""
    arr = np.zeros((N, 3))
    side = int(np.ceil(np.sqrt(N)))
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            arr[count, 0] = (i + 0.5) * w / side
            arr[count, 1] = (j + 0.5) * h / side
            arr[count, 2] = min(w, h) / (2 * side) * 0.85
            count += 1
    return arr


def _initial_random(w, h, seed=None):
    """Create random initial packing."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    arr = np.zeros((N, 3))
    for i in range(N):
        arr[i, 0] = random.uniform(0.1 * w, 0.9 * w)
        arr[i, 1] = random.uniform(0.1 * h, 0.9 * h)
        arr[i, 2] = random.uniform(0.05, 0.15) * min(w, h)
    
    return arr


def construct():
    start_time = time.time()
    
    best_arr = None
    best_sum = 0.0
    
    # Try different aspect ratios
    aspects = [1.0, 1.2, 0.8, 1.5, 0.67, 2.0, 0.5, 1.33, 0.75, 1.1, 0.9]
    
    for aspect in aspects:
        if time.time() - start_time > DEADLINE - 10.0:
            break
        
        h = 2.0 / (aspect + 1.0)
        w = aspect * h
        
        # Try multiple initializations
        for init_type in ['hex', 'grid', 'rand']:
            if time.time() - start_time > DEADLINE - 10.0:
                break
            
            if init_type == 'hex':
                arr_init = _initial_hexagonal(w, h)
            elif init_type == 'grid':
                arr_init = _initial_grid(w, h)
            else:
                arr_init = _initial_random(w, h, seed=random.randint(0, 10000))
            
            # Check if initial is valid
            if not _is_valid(arr_init, w, h):
                # Try to fix by shrinking
                for scale in [0.9, 0.8, 0.7, 0.6]:
                    arr_fixed = arr_init.copy()
                    arr_fixed[:, 2] *= scale
                    if _is_valid(arr_fixed, w, h):
                        arr_init = arr_fixed
                        break
                else:
                    continue
            
            # Expand
            arr_expanded = _expand_circles(arr_init, w, h)
            s = np.sum(arr_expanded[:, 2])
            
            # Polish
            arr_polished, s2 = _polish_result(arr_expanded, w, h, time_limit=3.0)
            if s2 > s:
                arr_expanded = arr_polished
                s = s2
            
            # Further refinement
            if time.time() - start_time < DEADLINE - 15.0:
                arr_final = _expansion_relaxation_polish(arr_expanded, w, h, time_limit=5.0)
                s3 = np.sum(arr_final[:, 2])
                if s3 > s:
                    arr_expanded = arr_final
                    s = s3
            
            if s > best_sum:
                best_sum = s
                best_arr = arr_expanded.copy()
    
    # Final fallback
    if best_arr is None or not _check_validity(best_arr):
        # Simple valid grid in a square
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
