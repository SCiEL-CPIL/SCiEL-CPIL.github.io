import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _compute_linear_overlaps(h):
    """Compute linear cross-correlation of h with (1-h) for all non-zero shifts."""
    g = 1.0 - h
    c = np.correlate(h, g, mode='full')
    # c has length 2N-1, center at N-1
    # Exclude zero shift
    c_neg = c[:N-1]
    c_pos = c[N:]
    return np.concatenate([c_neg, c_pos])

def _max_overlap(h):
    """Return the maximum linear overlap value."""
    overlaps = _compute_linear_overlaps(h)
    if len(overlaps) == 0:
        return 0.0
    return np.max(overlaps)

def _project_to_feasible(h):
    """Project h onto feasible set: 0 <= h_i <= 1 and sum(h) = N/2."""
    h = np.clip(h, 0, 1).copy()
    target = N / 2.0
    current = np.sum(h)
    
    if abs(current - target) < 1e-14:
        return h
    
    # Bisection on Lagrange multiplier
    lo, hi = -1.0, 1.0
    
    def sum_at_lambda(lam):
        return np.sum(np.clip(h - lam, 0, 1))
    
    for _ in range(80):
        mid = (lo + hi) / 2
        s = sum_at_lambda(mid)
        if abs(s - target) < 1e-14:
            break
        if s > target:
            lo = mid
        else:
            hi = mid
    
    lam = (lo + hi) / 2
    h = np.clip(h - lam, 0, 1)
    
    # Final adjustment
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 1e-10) & (h < 1 - 1e-10)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
    
    return h

def _objective_lse(h, beta):
    """LSE smoothed max overlap."""
    overlaps = _compute_linear_overlaps(h)
    if len(overlaps) == 0:
        return 0.0
    max_ov = np.max(overlaps)
    if beta == 0:
        return max_ov
    shifted = overlaps - max_ov
    shifted = np.clip(shifted, -50, 50)
    return max_ov + np.log(np.sum(np.exp(beta * shifted))) / beta

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Initialize with a good candidate from the previous logic (spectral/random)
    # We keep the initialization simple but diverse
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Try a few random starts and project
    for _ in range(10):
        h_cand = rng.uniform(0, 1, N)
        h_proj = _project_to_feasible(h_cand)
        score = _max_overlap(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj

    # Structural Change: Simulated Annealing / Metropolis-Hastings
    # We use a smooth approximation for the acceptance probability to avoid
    # the non-differentiability of the max, but evaluate the raw max for the score.
    
    T_start = 0.05
    T_end = 0.001
    num_steps = 20000
    
    current_h = best_h.copy()
    current_score = best_score
    
    # We will use a beta for the smooth objective in the acceptance criterion
    # to make the energy landscape smoother for the random walk
    beta = 20.0
    
    for step in range(num_steps):
        if time.time() > deadline - 5.0:
            break
            
        # Cooling schedule
        progress = step / num_steps
        T = T_start * (T_end / T_start) ** progress
        
        # Propose a move: Gaussian perturbation
        # Step size adapts to temperature
        step_size = 0.05 * (1.0 + 0.5 * np.sin(step * 0.01)) 
        delta = rng.normal(0, step_size, size=N)
        proposed_h = current_h + delta
        proposed_h = _project_to_feasible(proposed_h)
        
        # Evaluate
        # Use smooth objective for acceptance to allow uphill moves in the smooth landscape
        # but track the true max overlap for the "best" record
        current_smooth = _objective_lse(current_h, beta)
        proposed_smooth = _objective_lse(proposed_h, beta)
        
        delta_E = proposed_smooth - current_smooth
        
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            current_h = proposed_h
            current_score = _max_overlap(current_h)
            
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
                
    # Final polish with L-BFGS-B to tighten the solution found by SA
    if time.time() < deadline - 5.0:
        bounds = [(0.0, 1.0) for _ in range(N)]
        
        def obj_func(x):
            h = np.clip(x, 0, 1)
            return _objective_lse(h, 50.0)
        
        try:
            result = minimize(
                obj_func, best_h, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
            
    # Coordinate descent polish
    if time.time() < deadline - 2.0:
        improved = True
        max_cd = 20
        cd_iter = 0
        while improved and cd_iter < max_cd and time.time() < deadline - 2.0:
            improved = False
            cd_iter += 1
            for i in range(N):
                old_val = best_h[i]
                for delta in [0.02, -0.02, 0.01, -0.01, 0.005, -0.005]:
                    trial = best_h.copy()
                    trial[i] = np.clip(old_val + delta, 0, 1)
                    trial = _project_to_feasible(trial)
                    score = _max_overlap(trial)
                    if score < best_score - 1e-13:
                        best_h = trial
                        best_score = score
                        improved = True
                        break
                        
    best_h = _project_to_feasible(best_h)
    return best_h
