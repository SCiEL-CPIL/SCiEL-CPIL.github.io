import numpy as np
import time
import math
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _is_valid(centers, sides, tol=1e-7):
    """Check all constraints: inside unit square, positive sides, non-overlapping. Vectorized."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Inside unit square
    if np.any(centers - sides[:, None] / 2.0 < -tol):
        return False
    if np.any(centers + sides[:, None] / 2.0 > 1.0 + tol):
        return False
    # Non-overlap: for i < j, must not have |dx| < t AND |dy| < t where t = (s_i+s_j)/2 - tol
    dx = np.abs(centers[:, 0][:, None] - centers[:, 0][None, :])
    dy = np.abs(centers[:, 1][:, None] - centers[:, 1][None, :])
    t = (sides[:, None] + sides[None, :]) / 2.0 - tol
    # Only check upper triangle
    iu = np.triu_indices(n, k=1)
    if np.any((dx[iu] < t[iu]) & (dy[iu] < t[iu])):
        return False
    return True

def _objective(centers, sides):
    """Sum of side lengths."""
    return float(np.sum(sides))

def _max_side_at(cx, cy, centers, sides, i, tol=1e-7):
    """
    Compute the maximum side length for square i centered at (cx,cy)
    given the positions and sizes of all other squares. Vectorized.
    """
    # Bound from unit square
    max_s = min(2.0 * cx, 2.0 * (1.0 - cx), 2.0 * cy, 2.0 * (1.0 - cy))
    if max_s < 0:
        max_s = 0.0

    # For each other square j, compute the max s that avoids overlap:
    # s <= max(2*(|cx-cx_j|+tol) - s_j, 2*(|cy-cy_j|+tol) - s_j)
    # The overall bound is min over j != i of that max.
    mask = np.arange(len(sides)) != i
    dx = np.abs(cx - centers[mask, 0])
    dy = np.abs(cy - centers[mask, 1])
    sj = sides[mask]
    bounds = np.maximum(2.0 * (dx + tol) - sj, 2.0 * (dy + tol) - sj)
    if len(bounds) > 0:
        max_s = min(max_s, float(np.min(bounds)))

    if max_s < 0:
        max_s = 0.0
    return max_s

def _clamp_to_unit(centers, sides):
    """Clamp centers so squares stay inside [0,1]^2."""
    for i in range(len(sides)):
        half = sides[i] / 2.0
        centers[i, 0] = max(half, min(1.0 - half, centers[i, 0]))
        centers[i, 1] = max(half, min(1.0 - half, centers[i, 1]))
    return centers

def _resolve_overlap(centers, sides, tol=1e-7):
    """
    Attempt to resolve overlaps by pushing squares apart.
    """
    n = len(sides)
    for iteration in range(500):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2.0 - tol
                if t < 0:
                    continue
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                if abs(dx) < t and abs(dy) < t:
                    # Push apart
                    if abs(dx) < abs(dy):
                        push = (t - abs(dx)) / 2.0 + 2*tol
                        if dx >= 0:
                            centers[j, 0] += push
                            centers[i, 0] -= push
                        else:
                            centers[j, 0] -= push
                            centers[i, 0] += push
                    else:
                        push = (t - abs(dy)) / 2.0 + 2*tol
                        if dy >= 0:
                            centers[j, 1] += push
                            centers[i, 1] -= push
                        else:
                            centers[j, 1] -= push
                            centers[i, 1] += push
                    moved = True
        if not moved:
            break
    return centers

def _grow_all_squares(centers, sides, tol=1e-7, max_rounds=100):
    """
    Iteratively grow all squares to their maximum given current positions.
    Repeat until convergence.
    """
    n = len(sides)
    for _ in range(max_rounds):
        improved = False
        for i in range(n):
            new_s = _max_side_at(centers[i, 0], centers[i, 1], centers, sides, i, tol)
            if new_s > sides[i] + 1e-10:
                sides[i] = new_s
                improved = True
        if not improved:
            break
    return sides

def _grow_all_squares_fast(centers, sides, tol=1e-7, max_rounds=200):
    """
    Faster grow: compute all max sides in parallel each round using vectorized ops.
    """
    n = len(sides)
    for _ in range(max_rounds):
        improved = False
        new_sides = sides.copy()
        for i in range(n):
            new_sides[i] = _max_side_at(centers[i, 0], centers[i, 1], centers, sides, i, tol)
        if np.any(new_sides > sides + 1e-10):
            sides = new_sides
            improved = True
        if not improved:
            break
    return sides

def _smooth_penalty(centers, sides, tol=1e-7, beta=50.0):
    """
    Smooth surrogate for the nonsmooth overlap constraint using log-sum-exp.
    Returns a penalty that is ~0 when no overlap, and large when overlaps exist.
    """
    n = len(sides)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2.0 - tol
            if t < 0:
                continue
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            a = beta * (t - dx)
            b = beta * (t - dy)
            m = max(a, b)
            if m > 700:
                soft_min = m / beta
            else:
                soft_min = (m - math.log(math.exp(a - m) + math.exp(b - m))) / beta
            if soft_min > 0:
                penalty += soft_min ** 2
    return penalty

def _local_search(centers, sides, tol=1e-7, n_iter=200, rng=None, use_smooth=False, deadline=None):
    """
    Local search: perturb positions, then grow all squares.
    Keep the best result.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    n = len(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    best_obj = _objective(best_c, best_s)
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        # Perturb a random subset of squares
        k = int(rng.integers(1, min(n, 5)))
        idx = rng.choice(n, size=k, replace=False)
        trial_c = best_c.copy()
        for i in idx:
            step = rng.uniform(0.01, 0.08)
            trial_c[i, 0] += rng.uniform(-step, step)
            trial_c[i, 1] += rng.uniform(-step, step)
        
        trial_c = _clamp_to_unit(trial_c, best_s)
        
        if not _is_valid(trial_c, best_s, tol):
            # Try to resolve
            trial_c2 = trial_c.copy()
            trial_c2 = _resolve_overlap(trial_c2, best_s, tol)
            trial_c2 = _clamp_to_unit(trial_c2, best_s)
            if _is_valid(trial_c2, best_s, tol):
                trial_c = trial_c2
            else:
                continue
        
        # Grow all squares
        trial_s = best_s.copy()
        trial_s = _grow_all_squares(trial_c, trial_s, tol)
        
        if _is_valid(trial_c, trial_s, tol):
            obj = _objective(trial_c, trial_s)
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_c = trial_c.copy()
                best_s = trial_s.copy()
    
    return best_c, best_s

def _simulated_annealing(centers, sides, tol=1e-7, n_iter=2000, rng=None, 
                          t_start=0.5, t_end=0.001, deadline=None):
    """
    Simulated annealing with repulsion potential for temporary overlaps.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    n = len(sides)
    best_c = centers.copy()
    best_s = sides.copy()
    best_obj = _objective(best_c, best_s)
    
    # Current state (may be infeasible)
    curr_c = centers.copy()
    curr_s = sides.copy()
    
    # Temperature schedule
    cooling_rate = (t_end / t_start) ** (1.0 / max(n_iter - 1, 1))
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        T = t_start * (cooling_rate ** it)
        
        # Perturb: move a random subset of squares
        k = int(rng.integers(1, 4))
        idx = rng.choice(n, size=k, replace=False)
        trial_c = curr_c.copy()
        for i in idx:
            step = rng.uniform(0.02, 0.1)
            trial_c[i, 0] += rng.uniform(-step, step)
            trial_c[i, 1] += rng.uniform(-step, step)
        
        # Clamp to unit square
        trial_c = _clamp_to_unit(trial_c, curr_s)
        
        # Try to grow squares from this position
        trial_s = curr_s.copy()
        trial_s = _grow_all_squares(trial_c, trial_s, tol=tol, max_rounds=30)
        
        # Check validity
        if _is_valid(trial_c, trial_s, tol=tol):
            obj = _objective(trial_c, trial_s, tol=tol)
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_c = trial_c.copy()
                best_s = trial_s.copy()
            
            # Update current state
            curr_c = trial_c.copy()
            curr_s = trial_s.copy()
        else:
            # Infeasible: compute penalty and decide whether to accept
            curr_penalty = _smooth_penalty(curr_c, curr_s, tol=tol, beta=100.0)
            trial_penalty = _smooth_penalty(trial_c, trial_s, tol=tol, beta=100.0)
            
            # Also check if trial is valid after resolution
            trial_c2 = trial_c.copy()
            trial_c2 = _resolve_overlap(trial_c2, trial_s, tol=tol)
            trial_c2 = _clamp_to_unit(trial_c2, trial_s)
            if _is_valid(trial_c2, trial_s, tol=tol):
                obj = _objective(trial_c2, trial_s)
                if obj > best_obj + 1e-9:
                    best_obj = obj
                    best_c = trial_c2.copy()
                    best_s = trial_s.copy()
                curr_c = trial_c2.copy()
                curr_s = trial_s.copy()
            else:
                # Accept infeasible state with probability exp(-delta_penalty / T)
                delta = trial_penalty - curr_penalty
                if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / max(T, 1e-10))):
                    curr_c = trial_c.copy()
                    curr_s = trial_s.copy()
    
    return best_c, best_s

def _high_precision_polish(centers, sides, tol=1e-9, max_rounds=500, deadline=None):
    """
    Final high-precision polish: grow squares with tighter tolerance and more iterations.
    """
    best_c = centers.copy()
    best_s = sides.copy()
    best_obj = _objective(best_c, best_s)
    
    # First, grow with tight tolerance
    trial_s = best_s.copy()
    trial_s = _grow_all_squares(best_c, trial_s, tol=tol, max_rounds=max_rounds)
    if _is_valid(best_c, trial_s, tol=1e-7):
        obj = _objective(best_c, trial_s)
        if obj > best_obj:
            best_obj = obj
            best_s = trial_s.copy()
    
    # Small-step local search with tight tolerance
    rng = np.random.default_rng(99999)
    for it in range(150):
        if deadline is not None and time.time() > deadline:
            break
        k = int(rng.integers(1, 4))
        idx = rng.choice(10, size=k, replace=False)
        trial_c = best_c.copy()
        for i in idx:
            step = rng.uniform(0.001, 0.01)
            trial_c[i, 0] += rng.uniform(-step, step)
            trial_c[i, 1] += rng.uniform(-step, step)
        
        trial_c = _clamp_to_unit(trial_c, best_s)
        
        if not _is_valid(trial_c, best_s, tol=1e-7):
            trial_c2 = trial_c.copy()
            trial_c2 = _resolve_overlap(trial_c2, best_s, tol=1e-7)
            trial_c2 = _clamp_to_unit(trial_c2, best_s)
            if _is_valid(trial_c2, best_s, tol=1e-7):
                trial_c = trial_c2
            else:
                continue
        
        trial_s = best_s.copy()
        trial_s = _grow_all_squares(trial_c, trial_s, tol=tol, max_rounds=max_rounds)
        
        if _is_valid(trial_c, trial_s, tol=1e-7):
            obj = _objective(trial_c, trial_s)
            if obj > best_obj + 1e-10:
                best_obj = obj
                best_c = trial_c.copy()
                best_s = trial_s.copy()
    
    return best_c, best_s

def _init_symmetric_pattern1():
    """
    Pattern: 4 large squares in corners, 6 smaller filling the cross.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    
    s_large = 0.38
    centers[0] = [s_large/2, s_large/2]
    centers[1] = [1 - s_large/2, s_large/2]
    centers[2] = [s_large/2, 1 - s_large/2]
    centers[3] = [1 - s_large/2, 1 - s_large/2]
    sides[0:4] = s_large
    
    s_small = 0.20
    centers[4] = [0.5, s_large + s_small/2]
    centers[5] = [0.5, 1 - s_large - s_small/2]
    centers[6] = [s_large + s_small/2, 0.5]
    centers[7] = [1 - s_large - s_small/2, 0.5]
    centers[8] = [0.5, 0.5]
    sides[4:9] = s_small
    
    centers[9] = [0.5, 0.5 + s_small]
    sides[9] = s_small * 0.8
    
    return centers, sides

def _init_symmetric_pattern2():
    """
    2x5 grid: 2 rows of 5.
    Side = 1/5 = 0.2
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.full(10, 0.2, dtype=np.float64)
    idx = 0
    for row in range(2):
        for col in range(5):
            centers[idx, 0] = (col + 0.5) * 0.2
            centers[idx, 1] = (row + 0.5) * 0.5
            idx += 1
    return centers, sides

def _init_symmetric_pattern3():
    """
    One large (0.5) in bottom-left, 3x3 grid of small (1/6) in the L-shape.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    side_s = 1.0/6.0
    idx = 1
    for col in range(6):
        centers[idx, 0] = (col + 0.5) * side_s
        centers[idx, 1] = 0.5 + 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    for col in range(3):
        centers[idx, 0] = 0.5 + (col + 0.5) * side_s
        centers[idx, 1] = 0.5 * side_s
        sides[idx] = side_s
        idx += 1
    return centers, sides

def _init_symmetric_pattern4():
    """
    1 large (0.44) center, 8 around.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.5, 0.5]
    sides[0] = 0.44
    s_small = 0.28
    positions = [
        [0.14, 0.14], [0.86, 0.14], [0.14, 0.86], [0.86, 0.86],
        [0.5, 0.14], [0.5, 0.86], [0.14, 0.5], [0.86, 0.5]
    ]
    for i, pos in enumerate(positions):
        centers[i+1] = pos
        sides[i+1] = s_small
    return centers, sides

def _init_symmetric_pattern5():
    """
    5 large in a cross, 5 small in corners.
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)
    centers[0] = [0.5, 0.5]
    sides[0] = 0.25
    centers[1] = [0.5, 0.25]
    centers[2] = [0.5, 0.75]
    centers[3] = [0.25, 0.5]
    centers[4] = [0.75, 0.5]
    sides[1:5] = 0.25
    s_small = 0.15
    centers[5] = [0.125, 0.125]
    centers[6] = [0.875, 0.125]
    centers[7] = [0.125, 0.875]
    centers[8] = [0.875, 0.875]
    centers[9] = [0.5, 0.05]
    sides[5:10] = s_small
    return centers, sides

def _smooth_objective_and_grad(x, beta=100.0, lambda_pen=1000.0):
    """
    x: [cx1, cy1, s1, cx2, cy2, s2, ..., cx10, cy10, s10]
    Returns: (objective_value, gradient)
    """
    n = 10
    centers = x[0:20].reshape(10, 2)
    sides = x[20:30]
    
    # Objective: -sum(sides)
    obj = -np.sum(sides)
    
    # Gradient of objective
    grad = np.zeros_like(x)
    grad[20:30] = -1.0
    
    # Penalty for overlaps
    # For each pair (i, j), i < j
    # t = (s_i + s_j) / 2
    # dx = |cx_i - cx_j|
    # dy = |cy_i - cy_j|
    # a = t - dx
    # b = t - dy
    # soft_min = -1/beta * log(exp(-beta*a) + exp(-beta*b))
    # penalty = max(0, soft_min)^2
    
    # Penalty for box constraints
    # s_i/2 <= cx_i => s_i - 2*cx_i <= 0
    # s_i/2 <= 1-cx_i => s_i - 2*(1-cx_i) <= 0
    # s_i/2 <= cy_i => s_i - 2*cy_i <= 0
    # s_i/2 <= 1-cy_i => s_i - 2*(1-cy_i) <= 0
    
    # We compute the penalty and its gradient analytically.
    
    # To make it faster, we can vectorize the pair calculations.
    # But with n=10, there are 45 pairs. A loop is fine.
    
    for i in range(n):
        cx_i = centers[i, 0]
        cy_i = centers[i, 1]
        s_i = sides[i]
        
        # Box constraints for i
        # v1 = s_i - 2*cx_i
        # v2 = s_i - 2*(1-cx_i)
        # v3 = s_i - 2*cy_i
        # v4 = s_i - 2*(1-cy_i)
        # Penalty = sum (max(0, v_k)^2)
        # Grad: d/ds_i = 2*sum(max(0, v_k))
        #        d/dcx_i = -4*max(0, v1) + 4*max(0, v2)
        #        d/dcy_i = -4*max(0, v3) + 4*max(0, v4)
        
        v1 = s_i - 2*cx_i
        v2 = s_i - 2*(1-cx_i)
        v3 = s_i - 2*cy_i
        v4 = s_i - 2*(1-cy_i)
        
        p_box = 0.0
        g_s = 0.0
        g_cx = 0.0
        g_cy = 0.0
        
        if v1 > 0:
            p_box += v1**2
            g_s += 2*v1
            g_cx += -4*v1
        if v2 > 0:
            p_box += v2**2
            g_s += 2*v2
            g_cx += 4*v2
        if v3 > 0:
            p_box += v3**2
            g_s += 2*v3
            g_cy += -4*v3
        if v4 > 0:
            p_box += v4**2
            g_s += 2*v4
            g_cy += 4*v4
            
        obj += lambda_pen * p_box
        grad[2*i] += lambda_pen * g_cx
        grad[2*i+1] += lambda_pen * g_cy
        grad[20+i] += lambda_pen * g_s
        
    for i in range(n):
        for j in range(i+1, n):
            cx_i = centers[i, 0]
            cy_i = centers[i, 1]
            s_i = sides[i]
            cx_j = centers[j, 0]
            cy_j = centers[j, 1]
            s_j = sides[j]
            
            dx = abs(cx_i - cx_j)
            dy = abs(cy_i - cy_j)
            t = (s_i + s_j) / 2.0
            
            a = t - dx
            b = t - dy
            
            # soft_min(a, b)
            # m = max(a, b)
            # soft_min = (m - log(exp(a-m) + exp(b-m))) / beta
            # But we need max(0, soft_min)
            
            # If a <= 0 and b <= 0, soft_min <= 0, penalty = 0
            if a <= 0 and b <= 0:
                continue
                
            m = max(a, b)
            # Avoid overflow
            # exp(a-m) and exp(b-m) are in [0, 1]
            e_a = math.exp(a - m)
            e_b = math.exp(b - m)
            log_sum = math.log(e_a + e_b)
            soft_min_val = (m - log_sum) / beta
            
            if soft_min_val <= 0:
                continue
                
            p_overlap = soft_min_val ** 2
            obj += lambda_pen * p_overlap
            
            # Gradient of soft_min_val wrt a and b
            # d(soft_min)/da = exp(a-m)/(beta*(exp(a-m)+exp(b-m)))
            # d(soft_min)/db = exp(b-m)/(beta*(exp(a-m)+exp(b-m)))
            denom = beta * (e_a + e_b)
            d_smda = e_a / denom
            d_smdb = e_b / denom
            
            # d(p_overlap)/da = 2 * soft_min_val * d_smda
            # d(p_overlap)/db = 2 * soft_min_val * d_smdb
            
            grad_p_da = 2 * soft_min_val * d_smda
            grad_p_db = 2 * soft_min_val * d_smdb
            
            # a = t - dx = (s_i+s_j)/2 - |cx_i-cx_j|
            # da/ds_i = 0.5
            # da/ds_j = 0.5
            # da/dcx_i = -sign(cx_i - cx_j)
            # da/dcx_j = sign(cx_i - cx_j)
            
            sign_cx = 1.0 if cx_i >= cx_j else -1.0
            # dx = |cx_i - cx_j|, so d(dx)/dcx_i = sign(cx_i - cx_j)
            # da/dcx_i = - sign(cx_i - cx_j)
            
            grad[2*i] += lambda_pen * grad_p_da * (-sign_cx)
            grad[2*i+1] += lambda_pen * grad_p_db * (-sign_cy)
            grad[2*j] += lambda_pen * grad_p_da * (sign_cx)
            grad[2*j+1] += lambda_pen * grad_p_db * (sign_cy)
            grad[20+i] += lambda_pen * grad_p_da * 0.5 + lambda_pen * grad_p_db * 0.5
            grad[20+j] += lambda_pen * grad_p_da * 0.5 + lambda_pen * grad_p_db * 0.5
            
    return obj, grad

def _optimize_smooth(x0, deadline=None):
    """
    Run L-BFGS-B to minimize the smooth objective.
    """
    if deadline is not None and time.time() > deadline:
        return x0
        
    # Bounds: centers in [0, 1], sides in [0, 1]
    bounds = []
    for i in range(10):
        bounds.append((0.0, 1.0)) # cx
        bounds.append((0.0, 1.0)) # cy
        bounds.append((0.0, 1.0)) # s
        
    try:
        res = minimize(
            _smooth_objective_and_grad,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'gtol': 1e-8, 'maxls': 20}
        )
        return res.x
    except Exception:
        return x0

def squares_in_square10() -> tuple:
    """
    Maximize the sum of side lengths of 10 non-overlapping axis-aligned squares in the unit square.
    """
    start_time = time.time()
    deadline = start_time + 570.0
    
    inits = [
        _init_symmetric_pattern1(),
        _init_symmetric_pattern2(),
        _init_symmetric_pattern3(),
        _init_symmetric_pattern4(),
        _init_symmetric_pattern5(),
    ]
    
    best_centers = None
    best_sides = None
    best_obj = -1.0
    
    # Add some random inits
    rng_global = np.random.default_rng(12345)
    for _ in range(5):
        centers = rng_global.uniform(0.1, 0.9, size=(10, 2))
        sides = rng_global.uniform(0.1, 0.3, size=10)
        inits.append((centers, sides))
        
    for init_idx, (centers, sides) in enumerate(inits):
        if time.time() > deadline - 60:
            break
            
        centers = centers.copy()
        sides = sides.copy()
        
        # Ensure validity for local search/SA
        centers = _clamp_to_unit(centers, sides)
        if not _is_valid(centers, sides, tol=1e-7):
            centers = _resolve_overlap(centers, sides, tol=1e-7)
            centers = _clamp_to_unit(centers, sides)
            if not _is_valid(centers, sides, tol=1e-7):
                continue
        
        # Grow all squares
        sides = _grow_all_squares(centers, sides, tol=1e-7)
        if not _is_valid(centers, sides, tol=1e-7):
            continue
        
        obj = _objective(centers, sides)
        if obj > best_obj:
            best_obj = obj
            best_centers = centers.copy()
            best_sides = sides.copy()
            
        # Run smooth optimizer
        if time.time() < deadline - 30:
            # Flatten
            x0 = np.zeros(30)
            x0[0:20] = centers.flatten()
            x0[20:30] = sides
            
            # Run optimization
            x_opt = _optimize_smooth(x0, deadline=deadline - 30)
            
            # Unflatten
            c_opt = x_opt[0:20].reshape(10, 2)
            s_opt = x_opt[20:30]
            
            # Clamp and validate
            c_opt = _clamp_to_unit(c_opt, s_opt)
            # Grow to max
            s_opt = _grow_all_squares(c_opt, s_opt, tol=1e-7)
            
            if _is_valid(c_opt, s_opt, tol=1e-7):
                obj_opt = _objective(c_opt, s_opt)
                if obj_opt > best_obj:
                    best_obj = obj_opt
                    best_centers = c_opt.copy()
                    best_sides = s_opt.copy()
                    
        # Local search from current best
        if time.time() < deadline - 10 and best_centers is not None:
            rng = np.random.default_rng(init_idx * 7919 + 1)
            opt_c, opt_s = _local_search(best_centers, best_sides, tol=1e-7, n_iter=100, rng=rng, deadline=deadline - 10)
            if _is_valid(opt_c, opt_s, tol=1e-7):
                obj2 = _objective(opt_c, opt_s)
                if obj2 > best_obj:
                    best_obj = obj2
                    best_centers = opt_c.copy()
                    best_sides = opt_s.copy()
                    
    # If nothing valid, fallback
    if best_centers is None:
        centers, sides = _init_symmetric_pattern2()
        centers = _clamp_to_unit(centers, sides)
        sides = _grow_all_squares(centers, sides, tol=1e-7)
        best_centers = centers
        best_sides = sides
    else:
        # High-precision final polish with tighter tolerance
        if time.time() < deadline - 5:
            polished_c, polished_s = _high_precision_polish(best_centers, best_sides, tol=1e-9, max_rounds=500, deadline=deadline - 2)
            if _is_valid(polished_c, polished_s, tol=1e-7):
                obj_polished = _objective(polished_c, polished_s)
                if obj_polished > best_obj:
                    best_centers = polished_c
                    best_sides = polished_s
                    best_obj = obj_polished
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
