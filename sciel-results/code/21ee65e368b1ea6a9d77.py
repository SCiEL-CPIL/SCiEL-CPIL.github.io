"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
TARGET_SUM = N / 2.0

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _get_linear_overlaps(h):
    """
    Compute linear cross-correlation of h with (1-h) for all non-zero shifts.
    C_k = sum_i h_i * (1 - h_{i+k}) for shift k (positive and negative).
    """
    g = 1.0 - h
    c = np.correlate(h, g, mode='full')
    # c has length 2N-1, index N-1 is shift 0
    c_neg = c[:N-1]
    c_pos = c[N:]
    return np.concatenate([c_neg, c_pos])

def _max_overlap(h):
    """Return the maximum linear overlap value."""
    overlaps = _get_linear_overlaps(h)
    if len(overlaps) == 0:
        return 0.0
    return float(np.max(overlaps))

def _project_to_feasible(h):
    """
    Project h onto the feasible set: 0 <= h_i <= 1 and sum(h) = N/2.
    Uses bisection on a uniform shift for exact sum.
    """
    h = np.clip(h, 0.0, 1.0).copy()
    target = TARGET_SUM
    current = np.sum(h)
    
    if abs(current - target) < 1e-14:
        return h
    
    # Bisection to find delta such that sum(clip(h + delta, 0, 1)) = target
    lo, hi = -1.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        h_test = np.clip(h + mid, 0.0, 1.0)
        s = np.sum(h_test)
        if s < target:
            lo = mid
        else:
            hi = mid
        if abs(s - target) < 1e-14:
            break
    
    h = np.clip(h + (lo + hi) / 2.0, 0.0, 1.0)
    # Final exact correction
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    
    return h

def _objective_epigraph(x, beta, lam_sum):
    """
    Objective function using epigraph formulation.
    x = [h_0, ..., h_{N-1}, t]
    Minimize t + penalty
    Subject to: overlaps <= t (approximated via log-sum-exp)
    """
    h = np.clip(x[:N], 0.0, 1.0)
    t = x[N]
    
    overlaps = _get_linear_overlaps(h)
    
    if len(overlaps) == 0:
        lse = 0.0
    else:
        max_ov = np.max(overlaps)
        if max_ov > 1000:
            max_ov = 1000
        shifted = overlaps - max_ov
        shifted = np.clip(shifted, -50.0 / beta, 0.0)
        lse = max_ov + np.log(np.sum(np.exp(beta * shifted))) / beta
    
    # Penalize if lse > t
    if lse > t:
        penalty = (lse - t) ** 2
    else:
        penalty = 0.0
    
    sum_penalty = lam_sum * (np.sum(h) - TARGET_SUM) ** 2
    return t + penalty + sum_penalty

def _coordinate_descent_polish(h, max_iter=2000, tol=1e-10):
    """
    Coordinate descent local search to polish the solution.
    For each coordinate, try small perturbations and keep if it reduces max overlap.
    """
    h = h.copy()
    best_score = _max_overlap(h)
    
    for iteration in range(max_iter):
        improved = False
        for i in range(N):
            base = h[i]
            for delta in [0.01, -0.01, 0.005, -0.005, 0.001, -0.001]:
                new_val = base + delta
                if 0.0 <= new_val <= 1.0:
                    h[i] = new_val
                    h = _project_to_feasible(h)
                    score = _max_overlap(h)
                    if score < best_score - tol:
                        best_score = score
                        improved = True
                        break
            h[i] = base
        
        if not improved:
            for i in range(N):
                base = h[i]
                for delta in [0.001, -0.001, 0.0005, -0.0005, 0.0001, -0.0001]:
                    new_val = base + delta
                    if 0.0 <= new_val <= 1.0:
                        h[i] = new_val
                        h = _project_to_feasible(h)
                        score = _max_overlap(h)
                        if score < best_score - tol:
                            best_score = score
                            improved = True
                            break
                h[i] = base
            
            if not improved:
                break
    
    return h, best_score

def _pairwise_swap_polish(h, max_iter=500, tol=1e-10):
    """
    Pairwise swap local search: try swapping perturbations between pairs of coordinates.
    This can escape local minima that single-coordinate descent gets stuck in.
    """
    h = h.copy()
    best_score = _max_overlap(h)
    
    for iteration in range(max_iter):
        improved = False
        for i in range(N):
            for j in range(i + 1, N):
                base_i, base_j = h[i], h[j]
                # Try moving value from i to j and vice versa
                for delta in [0.01, -0.01, 0.005, -0.005]:
                    new_i = base_i + delta
                    new_j = base_j - delta
                    if 0.0 <= new_i <= 1.0 and 0.0 <= new_j <= 1.0:
                        h[i] = new_i
                        h[j] = new_j
                        h = _project_to_feasible(h)
                        score = _max_overlap(h)
                        if score < best_score - tol:
                            best_score = score
                            improved = True
                            break
                h[i] = base_i
                h[j] = base_j
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return h, best_score

def construct():
    """
    Construct the step heights h to minimize the maximum overlap of h with shifted 1-h.
    
    Uses epigraph formulation with auxiliary variable t for smoother optimization,
    followed by coordinate descent and pairwise swap polish.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    best_h = None
    best_score = float('inf')
    
    # Multiple structured initial guesses
    t_arr = np.arange(N)
    
    # Initial guess 1: sinusoidal freq 3
    h_init = np.full(N, 0.5)
    h_init += 0.3 * np.sin(2 * np.pi * 3 * t_arr / N)
    h_init = _project_to_feasible(h_init)
    score = _max_overlap(h_init)
    if score < best_score:
        best_score = score
        best_h = h_init.copy()
    
    # Initial guess 2: sinusoidal freq 7
    h_init2 = np.full(N, 0.5)
    h_init2 += 0.3 * np.sin(2 * np.pi * 7 * t_arr / N)
    h_init2 = _project_to_feasible(h_init2)
    score2 = _max_overlap(h_init2)
    if score2 < best_score:
        best_score = score2
        best_h = h_init2.copy()
    
    # Initial guess 3: random
    h_init3 = rng.uniform(0, 1, N)
    h_init3 = _project_to_feasible(h_init3)
    score3 = _max_overlap(h_init3)
    if score3 < best_score:
        best_score = score3
        best_h = h_init3.copy()
    
    # Initial guess 4: alternating pattern
    h_init4 = np.zeros(N)
    h_init4[::2] = 0.8
    h_init4[1::2] = 0.2
    h_init4 = _project_to_feasible(h_init4)
    score4 = _max_overlap(h_init4)
    if score4 < best_score:
        best_score = score4
        best_h = h_init4.copy()
    
    # Initial guess 5: sinusoidal freq 1
    h_init5 = np.full(N, 0.5)
    h_init5 += 0.3 * np.sin(2 * np.pi * 1 * t_arr / N)
    h_init5 = _project_to_feasible(h_init5)
    score5 = _max_overlap(h_init5)
    if score5 < best_score:
        best_score = score5
        best_h = h_init5.copy()
    
    # Initial guess 6: sinusoidal freq 5
    h_init6 = np.full(N, 0.5)
    h_init6 += 0.3 * np.sin(2 * np.pi * 5 * t_arr / N)
    h_init6 = _project_to_feasible(h_init6)
    score6 = _max_overlap(h_init6)
    if score6 < best_score:
        best_score = score6
        best_h = h_init6.copy()
    
    t_init = best_score + 0.1
    bounds_h = [(0.0, 1.0) for _ in range(N)]
    bounds_t = [(0.0, 10.0)]
    bounds = bounds_h + bounds_t
    
    # Phase 1: Coarse optimization with increasing beta
    beta_schedule = [10, 30, 100, 300, 1000]
    
    for beta in beta_schedule:
        if time.time() > deadline - 30.0:
            break
        
        x_init = np.concatenate([best_h, [t_init]])
        
        def obj_func(x):
            return _objective_epigraph(x, beta, 50.0)
        
        try:
            result = minimize(
                obj_func, x_init, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30}
            )
            h_new = _project_to_feasible(result.x[:N])
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
                t_init = result.x[N]
        except Exception:
            pass
    
    # Phase 2: High-precision polish
    if time.time() < deadline - 20.0:
        x_init = np.concatenate([best_h, [t_init]])
        
        def obj_polish(x):
            return _objective_epigraph(x, 5000.0, 5000.0)
        
        try:
            result = minimize(
                obj_polish, x_init, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            h_polished = _project_to_feasible(result.x[:N])
            score_polished = _max_overlap(h_polished)
            if score_polished < best_score:
                best_score = score_polished
                best_h = h_polished
        except Exception:
            pass
    
    # Phase 3: Coordinate descent polish
    if time.time() < deadline - 10.0:
        h_cd, score_cd = _coordinate_descent_polish(best_h, max_iter=1000, tol=1e-10)
        if score_cd < best_score:
            best_score = score_cd
            best_h = h_cd
    
    # Phase 4: Pairwise swap polish
    if time.time() < deadline - 5.0:
        h_sw, score_sw = _pairwise_swap_polish(best_h, max_iter=200, tol=1e-10)
        if score_sw < best_score:
            best_score = score_sw
            best_h = h_sw
    
    # Final coordinate descent after swaps
    if time.time() < deadline - 2.0:
        h_cd2, score_cd2 = _coordinate_descent_polish(best_h, max_iter=500, tol=1e-11)
        if score_cd2 < best_score:
            best_score = score_cd2
            best_h = h_cd2
    
    # Ensure final validity
    best_h = _project_to_feasible(best_h)
    
    # Final precision polish: ensure sum is exactly N/2
    sum_h = np.sum(best_h)
    diff = TARGET_SUM - sum_h
    if abs(diff) > 1e-12:
        if best_h[0] + diff <= 1.0 and best_h[0] + diff >= 0.0:
            best_h[0] += diff
        else:
            best_h = best_h * (TARGET_SUM / sum_h)
            best_h = np.clip(best_h, 0.0, 1.0)
            best_h = _project_to_feasible(best_h)
    
    return best_h

# EVOLVE-BLOCK-END
