"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 circles."""
import numpy as np
import time
from scipy.optimize import minimize
import random

N = 21
DEADLINE = 145.0  # seconds wall-clock limit (evaluator kills at 150s, use 145s safety)
SAFETY_MARGIN = 5.0


def _is_valid(arr, tol=1e-9):
    """Check if circles are valid: disjoint interiors, inside bounding box, perimeter <= 4."""
    n = arr.shape[0]
    if n != N:
        return False
    xs, ys, rs = arr[:, 0], arr[:, 1], arr[:, 2]
    
    if np.any(rs <= 0):
        return False
    
    # Bounding box extents
    x_ext_min = np.min(xs - rs)
    x_ext_max = np.max(xs + rs)
    y_ext_min = np.min(ys - rs)
    y_ext_max = np.max(ys + rs)
    
    perim = 2.0 * ((x_ext_max - x_ext_min) + (y_ext_max - y_ext_min))
    if perim > 4.0 + tol:
        return False
    
    # Disjoint interiors - vectorized
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = rs[i_idx] + rs[j_idx]
    if np.any(dist < min_dist - tol):
        return False
    
    return True


def _logsumexp_penalty(violations, temperature=0.01):
    """Smooth log-sum-exp approximation of sum of positive violations."""
    # violations is an array of non-negative values
    if len(violations) == 0:
        return 0.0
    # Use log-sum-exp for numerical stability
    v = np.maximum(violations, 0.0)
    if np.all(v < 1e-12):
        return 0.0
    v_scaled = v / temperature
    max_v = np.max(v_scaled)
    if max_v > 500:  # avoid overflow
        return np.sum(v)  # fall back to linear sum
    log_sum = max_v + np.log(np.sum(np.exp(v_scaled - max_v)))
    return temperature * log_sum


def _vectorized_objective(vars_):
    """Negative sum of radii, with smooth penalty for constraint violations using log-sum-exp."""
    n = len(vars_) // 3
    x = vars_[::3]
    y = vars_[1::3]
    r = vars_[2::3]
    
    # Bounding box
    x_min = np.min(x - r)
    x_max = np.max(x + r)
    y_min = np.min(y - r)
    y_max = np.max(y + r)
    
    w = x_max - x_min
    h = y_max - y_min
    
    # Perimeter constraint: w + h <= 2
    perim_viol = max(0.0, (w + h) - 2.0)
    
    # Overlap constraints - vectorized
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.sqrt(dx * dx + dy * dy)
    overlap_violations = np.maximum(r[i_idx] + r[j_idx] - dist, 0.0)
    
    # Smooth penalty using log-sum-exp
    penalty = 50.0 * (perim_viol ** 2 + _logsumexp_penalty(overlap_violations))
    return -np.sum(r) + penalty


def _constraint_refine(arr, max_iter=500, tol=1e-10):
    """Refine positions and radii to satisfy constraints more tightly."""
    n = arr.shape[0]
    best = arr.copy()
    best_val = np.sum(best[:, 2])
    
    for it in range(max_iter):
        # Try to increase a random radius slightly
        i = random.randint(0, n - 1)
        dr = random.uniform(0.001, 0.01)
        arr_new = best.copy()
        arr_new[i, 2] += dr
        
        if _is_valid(arr_new):
            best = arr_new
            best_val = np.sum(best[:, 2])
        else:
            # Try to move the circle to make room
            for _ in range(10):
                dx = random.uniform(-0.05, 0.05)
                dy = random.uniform(-0.05, 0.05)
                arr_try = best.copy()
                arr_try[i, 0] += dx
                arr_try[i, 1] += dy
                arr_try[i, 2] = best[i, 2] + dr
                if _is_valid(arr_try):
                    best = arr_try
                    best_val = np.sum(best[:, 2])
                    break
    return best, best_val


def _force_relax_unified(arr, max_iter=1000):
    """Force relaxation to handle constraint violations."""
    n = arr.shape[0]
    best = arr.copy()
    
    for it in range(max_iter):
        # Check for violations - vectorized
        xs, ys, rs = best[:, 0], best[:, 1], best[:, 2]
        x_min = np.min(xs - rs)
        x_max = np.max(xs + rs)
        y_min = np.min(ys - rs)
        y_max = np.max(ys + rs)
        w = x_max - x_min
        h = y_max - y_min
        perim_viol = max(0.0, (w + h) - 2.0)
        
        # Vectorized overlap check
        i_idx, j_idx = np.triu_indices(n, k=1)
        dx = xs[i_idx] - xs[j_idx]
        dy = ys[i_idx] - ys[j_idx]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap_violations = np.maximum(rs[i_idx] + rs[j_idx] - dist, 0.0)
        overlap_viol = np.sum(overlap_violations)
        
        if perim_viol < 1e-6 and overlap_viol < 1e-6:
            break
        
        # Reduce radii slightly to fix violations
        best[:, 2] *= 0.99
        
    return best


def _expansion_relaxation_polish(arr, max_iter=2000):
    """Polish by expanding radii where possible."""
    n = arr.shape[0]
    best = arr.copy()
    best_val = np.sum(best[:, 2])
    
    for it in range(max_iter):
        # Try to expand each circle
        for i in range(n):
            # For now, just try small expansion
            dr = 0.005
            arr_try = best.copy()
            arr_try[i, 2] += dr
            
            if _is_valid(arr_try):
                best = arr_try
                best_val = np.sum(best[:, 2])
    
    return best, best_val


def _initial_hexagonal(w, h):
    """Create hexagonal packing initial guess."""
    arr = np.zeros((N, 3))
    # Hexagonal grid
    dx = 0.18
    dy = dx * np.sqrt(3) / 2
    count = 0
    for row in range(10):
        for col in range(10):
            if count >= N:
                break
            x_off = dx / 2 if row % 2 == 1 else 0
            x = (col + 0.5) * dx + x_off
            y = (row + 0.5) * dy
            if x < w and y < h:
                arr[count, 0] = x
                arr[count, 1] = y
                arr[count, 2] = dx / 2 * 0.9
                count += 1
        if count >= N:
            break
    return arr


def _initial_grid(w, h):
    """Create grid initial guess."""
    arr = np.zeros((N, 3))
    side = 5
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


def _initial_random(w, h):
    """Create random initial guess."""
    arr = np.random.rand(N, 3)
    arr[:, 0] *= w
    arr[:, 1] *= h
    arr[:, 2] = 0.05 + np.random.rand(N) * 0.05
    return arr


def _polish_result(arr, time_budget=10.0):
    """Polish the result using multiple techniques."""
    start = time.time()
    best = arr.copy()
    best_val = np.sum(best[:, 2])
    
    # Step 1: Force relaxation
    if time.time() - start < time_budget * 0.2:
        relaxed = _force_relax_unified(best, max_iter=200)
        if np.sum(relaxed[:, 2]) > best_val:
            best = relaxed
            best_val = np.sum(best[:, 2])
    
    # Step 2: Constraint refinement
    if time.time() - start < time_budget * 0.5:
        refined, val = _constraint_refine(best, max_iter=300)
        if val > best_val:
            best = refined
            best_val = val
    
    # Step 3: Expansion polish
    if time.time() - start < time_budget * 0.9:
        polished, val = _expansion_relaxation_polish(best, max_iter=500)
        if val > best_val:
            best = polished
            best_val = val
    
    return best, best_val


def _max_radius_for_circle(arr, i):
    """Compute max radius for circle i given others."""
    n = arr.shape[0]
    x_i, y_i, r_i = arr[i]
    xs, ys, rs = arr[:, 0], arr[:, 1], arr[:, 2]
    
    max_r = 1.0  # large initial guess
    
    # Distance to other circles
    for j in range(n):
        if j == i:
            continue
        dx = x_i - xs[j]
        dy = y_i - ys[j]
        dist = np.sqrt(dx * dx + dy * dy)
        max_r = min(max_r, max(0.0, dist - rs[j]))
    
    return max_r


def _is_valid_move(arr, i, dx, dy, dr):
    """Check if a move is valid."""
    arr_try = arr.copy()
    arr_try[i, 0] += dx
    arr_try[i, 1] += dy
    arr_try[i, 2] = max(1e-6, arr_try[i, 2] + dr)
    return _is_valid(arr_try)


def _slsqp_refine(arr, w, h, max_iter=200):
    """Use SLSQP optimization to refine positions and radii within a fixed rectangle."""
    n = arr.shape[0]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = arr[i, 0]
        x0[3*i+1] = arr[i, 1]
        x0[3*i+2] = arr[i, 2]
    
    def obj(vars_):
        return _vectorized_objective(vars_)
    
    def gradient(vars_):
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(vars_)
        for i in range(len(vars_)):
            vars_plus = vars_.copy()
            vars_plus[i] += eps
            vars_minus = vars_.copy()
            vars_minus[i] -= eps
            grad[i] = (obj(vars_plus) - obj(vars_minus)) / (2 * eps)
        return grad
    
    # Constraints: perimeter and non-overlap
    constraints = []
    
    # Perimeter constraint: 2*(w + h) <= 4, i.e., w + h <= 2
    # But w and h are determined by the circles, so we need to express in terms of variables
    # Actually, the bounding box is determined by the circles, so we can't easily constrain it
    # Let's use a penalty approach instead
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, w))  # x position
        bounds.append((0.0, h))  # y position
        bounds.append((0.0, min(w, h) / 2))  # radius
    
    try:
        result = minimize(
            obj, x0, jac=gradient,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-10}
        )
        
        if result.success:
            best_vars = result.x
            arr_out = np.zeros((n, 3))
            for i in range(n):
                arr_out[i, 0] = best_vars[3*i]
                arr_out[i, 1] = best_vars[3*i+1]
                arr_out[i, 2] = best_vars[3*i+2]
            
            if _is_valid(arr_out):
                return arr_out
    except Exception:
        pass
    
    return arr


def construct():
    start_time = time.time()
    
    best_arr = None
    best_sum = 0.0
    
    # Try different aspect ratios
    aspects = [1.0, 1.2, 0.8, 1.5, 0.67, 2.0, 0.5, 1.33, 0.75, 1.1, 0.9]
    
    for aspect in aspects:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        h = 2.0 / (aspect + 1.0)
        w = aspect * h
        
        # Generate initial configurations
        inits = [
            _initial_hexagonal(w, h),
            _initial_grid(w, h),
            _initial_random(w, h),
        ]
        
        for arr_init in inits:
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            # SLSQP refinement
            if time.time() - start_time < DEADLINE - SAFETY_MARGIN:
                refined = _slsqp_refine(arr_init, w, h, max_iter=100)
                if _is_valid(refined):
                    val = np.sum(refined[:, 2])
                    if val > best_sum:
                        best_sum = val
                        best_arr = refined.copy()
            
            # Polish
            if time.time() - start_time < DEADLINE - SAFETY_MARGIN:
                polished, val = _polish_result(arr_init, time_budget=3.0)
                if val > best_sum:
                    best_sum = val
                    best_arr = polished.copy()
    
    # Final polish with remaining time
    if best_arr is not None and time.time() - start_time < DEADLINE - 2.0:
        remaining = DEADLINE - 2.0 - (time.time() - start_time)
        if remaining > 1.0:
            polished, val = _polish_result(best_arr, time_budget=remaining)
            if val > best_sum:
                best_sum = val
                best_arr = polished
    
    if best_arr is None:
        # Fallback
        side = 5
        positions = [(i, j) for i in range(side) for j in range(side)][:N]
        best_arr = np.zeros((N, 3))
        for idx, (i, j) in enumerate(positions):
            best_arr[idx, 0] = (i + 0.5) / side
            best_arr[idx, 1] = (j + 0.5) / side
            best_arr[idx, 2] = 0.5 / side - 0.01
    
    if not _is_valid(best_arr):
        # Return a known-valid fallback
        side = 5
        positions = [(i, j) for i in range(side) for j in range(side)][:N]
        best_arr = np.zeros((N, 3))
        for idx, (i, j) in enumerate(positions):
            best_arr[idx, 0] = (i + 0.5) / side
            best_arr[idx, 1] = (j + 0.5) / side
            best_arr[idx, 2] = 0.5 / side - 0.01
    
    return best_arr
