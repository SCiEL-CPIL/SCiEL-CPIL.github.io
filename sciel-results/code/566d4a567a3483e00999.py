"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
import random
from scipy.optimize import minimize, differential_evolution

N = 21
DEADLINE = 150.0  # seconds wall-clock limit (evaluator kills at 150s, keep margin)


def _is_valid(arr, w, h, tol=1e-9):
    """Check if circles are valid in rectangle [0,w] x [0,h]."""
    xs, ys, rs = arr[:, 0], arr[:, 1], arr[:, 2]
    
    if np.any(rs <= 0):
        return False
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


def _expand_radii(arr, w, h, max_iters=500, tol=1e-12):
    """Iteratively expand each circle to its maximum possible radius."""
    arr = arr.copy()
    for _ in range(max_iters):
        improved = False
        for i in range(N):
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
                r_max = min(r_max, dist - arr[j, 2])
            r_max = max(r_max, 0.0)
            if r_max > arr[i, 2] + tol:
                arr[i, 2] = r_max
                improved = True
        if not improved:
            break
    return arr


def _objective(z, w, h):
    """
    Objective for optimization: minimize negative sum of radii subject to constraints.
    z = [x0, y0, r0, x1, y1, r1, ..., x20, y20, r20]
    We use a penalty method: minimize -sum(r) + penalty * violations.
    """
    xs = z[0::3]
    ys = z[1::3]
    rs = z[2::3]
    
    # Penalty for boundary violations
    penalty = 0.0
    for i in range(N):
        if rs[i] <= 0:
            penalty += 10.0
        if xs[i] - rs[i] < 0:
            penalty += 10.0 * (xs[i] - rs[i])
        if xs[i] + rs[i] > w:
            penalty += 10.0 * (xs[i] + rs[i] - w)
        if ys[i] - rs[i] < 0:
            penalty += 10.0 * (ys[i] - rs[i])
        if ys[i] + rs[i] > h:
            penalty += 10.0 * (ys[i] + rs[i] - h)
    
    # Penalty for pairwise overlap
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = rs[i] + rs[j] - dist
            if overlap > 0:
                penalty += 10.0 * overlap
    
    return -np.sum(rs) + penalty


def _vectorized_objective(z, w, h):
    """Vectorized objective for faster evaluation."""
    xs = z[0::3]
    ys = z[1::3]
    rs = z[2::3]
    
    # Boundary penalties
    pen_x_low = np.maximum(0, rs - xs)
    pen_x_high = np.maximum(0, xs + rs - w)
    pen_y_low = np.maximum(0, rs - ys)
    pen_y_high = np.maximum(0, ys + rs - h)
    penalty = 10.0 * (np.sum(pen_x_low) + np.sum(pen_x_high) + np.sum(pen_y_low) + np.sum(pen_y_high))
    
    # Pairwise penalties (vectorized)
    for i in range(N):
        dx = xs[i] - xs[i+1:]
        dy = ys[i] - ys[i+1:]
        dists = np.sqrt(dx*dx + dy*dy)
        overlaps = rs[i] + rs[i+1:] - dists
        penalty += 10.0 * np.sum(np.maximum(0, overlaps))
    
    return -np.sum(rs) + penalty


def _refine_positions(arr, w, h, time_limit=10.0):
    """
    Refine positions and radii using scipy minimize with a smooth objective.
    We optimize all 3*N variables (x, y, r for each circle).
    """
    start = time.time()
    
    z0 = np.zeros(3 * N)
    for i in range(N):
        z0[3*i] = arr[i, 0]
        z0[3*i+1] = arr[i, 1]
        z0[3*i+2] = arr[i, 2]
    
    bounds = []
    for i in range(N):
        bounds.append((0, w))       # x
        bounds.append((0, h))       # y
        bounds.append((0, min(w, h)/2))  # r
    
    try:
        result = minimize(_vectorized_objective, z0, args=(w, h),
                         method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if time.time() - start > time_limit:
            return arr
        
        z_opt = result.x
        arr_new = np.zeros((N, 3))
        for i in range(N):
            arr_new[i, 0] = z_opt[3*i]
            arr_new[i, 1] = z_opt[3*i+1]
            arr_new[i, 2] = max(z_opt[3*i+2], 0.01)
        
        if _is_valid(arr_new, w, h):
            arr_expanded = _expand_radii(arr_new, w, h)
            return arr_expanded
    except Exception:
        pass
    
    return arr


def _pack_single(w, h, time_budget=15.0, seed=None):
    """Pack circles in a specific rectangle using multiple strategies."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    start = time.time()
    best_arr = None
    best_sum = 0.0
    
    # Strategy 1: Grid initialization + expansion + refinement
    arr_grid = np.zeros((N, 3))
    side = 5  # 5x5 grid, use 21 of 25
    count = 0
    for i in range(5):
        for j in range(5):
            if count >= N:
                break
            arr_grid[count, 0] = (i + 0.5) * w / 5
            arr_grid[count, 1] = (j + 0.5) * h / 5
            arr_grid[count, 2] = min(w, h) / (2 * 5) * 0.7
            count += 1
    
    if _is_valid(arr_grid, w, h):
        arr_exp = _expand_radii(arr_grid, w, h)
        s = np.sum(arr_exp[:, 2])
        if s > best_sum:
            best_sum = s
            best_arr = arr_exp.copy()
    
    # Strategy 2: Hexagonal-like initialization
    arr_hex = np.zeros((N, 3))
    r_est = min(w, h) / 6.0 * 0.8
    idx = 0
    row = 0
    while idx < N and row < 10:
        col = 0
        while idx < N and col < 10:
            offset = 0.5 * r_est * np.sqrt(3) if row % 2 else 0
            x = (col + 0.5) * r_est * np.sqrt(3) + offset
            y = (row + 0.5) * r_est * 1.5
            if x + r_est <= w and y + r_est <= h:
                arr_hex[idx, 0] = x
                arr_hex[idx, 1] = y
                arr_hex[idx, 2] = r_est
                idx += 1
            col += 1
        row += 1
    
    if idx == N and _is_valid(arr_hex, w, h):
        arr_exp = _expand_radii(arr_hex, w, h)
        s = np.sum(arr_exp[:, 2])
        if s > best_sum:
            best_sum = s
            best_arr = arr_exp.copy()
    
    # Strategy 3: Random initialization + expansion
    for attempt in range(3):
        if time.time() - start > time_budget * 0.5:
            break
        
        arr_rand = np.zeros((N, 3))
        for i in range(N):
            arr_rand[i, 0] = random.uniform(0.1 * w, 0.9 * w)
            arr_rand[i, 1] = random.uniform(0.1 * h, 0.9 * h)
            arr_rand[i, 2] = random.uniform(0.05, 0.12) * min(w, h)
        
        # Shrink if invalid
        if not _is_valid(arr_rand, w, h):
            for scale in [0.8, 0.6, 0.5]:
                arr_rand[:, 2] *= scale
                if _is_valid(arr_rand, w, h):
                    break
            else:
                continue
        
        arr_exp = _expand_radii(arr_rand, w, h)
        s = np.sum(arr_exp[:, 2])
        if s > best_sum:
            best_sum = s
            best_arr = arr_exp.copy()
    
    # Refine best solution
    if best_arr is not None and time.time() - start < time_budget * 0.7:
        arr_refined = _refine_positions(best_arr, w, h, time_limit=time_budget * 0.25)
        s = np.sum(arr_refined[:, 2])
        if s > best_sum:
            best_sum = s
            best_arr = arr_refined.copy()
    
    # Additional local search: perturb and re-expand
    if best_arr is not None and time.time() - start < time_budget:
        arr_current = best_arr.copy()
        step_size = 0.01 * min(w, h)
        for _ in range(50):
            if time.time() - start > time_budget:
                break
            i = random.randint(0, N - 1)
            dx = random.uniform(-step_size, step_size)
            dy = random.uniform(-step_size, step_size)
            
            arr_new = arr_current.copy()
            arr_new[i, 0] = np.clip(arr_new[i, 0] + dx, 0, w)
            arr_new[i, 1] = np.clip(arr_new[i, 1] + dy, 0, h)
            
            if _is_valid(arr_new, w, h):
                arr_exp = _expand_radii(arr_new, w, h)
                s = np.sum(arr_exp[:, 2])
                if s > best_sum + 1e-10:
                    best_sum = s
                    best_arr = arr_exp.copy()
                    arr_current = arr_exp.copy()
                    step_size *= 0.95
                else:
                    step_size *= 0.98
    
    return best_arr, best_sum


def construct():
    start_time = time.time()
    
    best_arr = None
    best_sum = 0.0
    
    # Try different aspect ratios (w/h)
    # Perimeter = 2(w+h) = 4, so w+h = 2
    aspects = [1.0, 1.2, 0.8, 1.5, 2/3, 2.0, 0.5, 4/3, 3/4, 1.1, 0.9, 1.3, 0.77]
    
    for aspect in aspects:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15.0:
            break
        
        h = 2.0 / (aspect + 1.0)
        w = aspect * h
        
        # Budget per aspect ratio
        time_budget = min(12.0, (DEADLINE - 15.0) - elapsed)
        if time_budget < 3.0:
            continue
        
        arr, s = _pack_single(w, h, time_budget=time_budget, seed=random.randint(0, 100000))
        
        if arr is not None and s > best_sum:
            best_sum = s
            best_arr = arr.copy()
    
    # Final verification and fallback
    if best_arr is None or not _is_valid(best_arr, 0, 0) or best_sum < 1.0:
        # Simple valid grid in a square
        w = h = 1.0
        best_arr = np.zeros((N, 3))
        count = 0
        for i in range(5):
            for j in range(5):
                if count >= N:
                    break
                best_arr[count, 0] = (i + 0.5) / 5
                best_arr[count, 1] = (j + 0.5) / 5
                best_arr[count, 2] = 0.5 / 5 - 0.02
                count += 1
        best_sum = np.sum(best_arr[:, 2])
    
    # Final check: ensure validity
    # Compute bounding box
    xs, ys, rs = best_arr[:, 0], best_arr[:, 1], best_arr[:, 2]
    x_min = np.min(xs - rs)
    x_max = np.max(xs + rs)
    y_min = np.min(ys - rs)
    y_max = np.max(ys + rs)
    perim = 2.0 * ((x_max - x_min) + (y_max - y_min))
    
    if perim > 4.0 + 1e-9:
        # Scale down to fit
        scale = 4.0 / perim * 0.999
        best_arr[:, 0] = x_min + (best_arr[:, 0] - x_min) * scale
        best_arr[:, 1] = y_min + (best_arr[:, 1] - y_min) * scale
        best_arr[:, 2] *= scale
    
    return best_arr
