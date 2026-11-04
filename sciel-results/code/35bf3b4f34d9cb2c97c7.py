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
    Returns an array of overlap values C_k for k != 0.
    C_k = sum_i h_i * (1 - h_{i+k})
    """
    g = 1.0 - h
    # Full correlation: result[0..2N-2]
    # np.correlate(h, g, 'full') computes sum_i h[i] * g[i + k - (N-1)]
    # The center (zero shift) is at index N-1.
    c = np.correlate(h, g, mode='full')
    # We exclude the zero-shift correlation c[N-1]
    # Negative shifts: c[0 : N-1]
    # Positive shifts: c[N : 2N-1]
    c_neg = c[:N-1]
    c_pos = c[N:]
    return np.concatenate([c_neg, c_pos])

def _max_overlap(h):
    """Return the maximum linear overlap value."""
    overlaps = _get_linear_overlaps(h)
    if len(overlaps) == 0:
        return 0.0
    return np.max(overlaps)

def _project_to_feasible(h):
    """
    Project h onto the feasible set: 0 <= h_i <= 1 and sum(h) = N/2.
    Uses iterative water-filling style projection.
    """
    h = np.clip(h, 0, 1).copy()
    target = TARGET_SUM
    current = np.sum(h)
    
    if abs(current - target) < 1e-12:
        return h
    
    for _ in range(100):
        diff = target - np.sum(h)
        if abs(diff) < 1e-12:
            break
        
        if diff > 0:
            # Need to increase sum. Add to elements < 1.
            mask = h < 1.0
            if not np.any(mask):
                break
            count = np.sum(mask)
            # Distribute diff proportionally to available room, or just evenly?
            # Even distribution with clipping is simpler and robust for this convex set.
            step = diff / count
            h[mask] += step
            h = np.clip(h, 0, 1)
        else:
            # Need to decrease sum. Subtract from elements > 0.
            mask = h > 0.0
            if not np.any(mask):
                break
            count = np.sum(mask)
            step = (-diff) / count
            h[mask] -= step
            h = np.clip(h, 0, 1)
    
    return h

def _objective_lse(x, beta, lam_sum):
    """
    Objective function using LogSumExp approximation of the max overlap.
    """
    h = np.clip(x, 0, 1)
    overlaps = _get_linear_overlaps(h)
    
    if len(overlaps) == 0:
        max_ov = 0.0
    else:
        max_ov = np.max(overlaps)
    
    # Smooth max
    if max_ov > 1000:
        max_ov = 1000
        
    if beta > 0:
        # Shift for stability
        shifted = overlaps - max_ov
        # Use log-sum-exp trick: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
        lse = max_ov + np.log(np.sum(np.exp(beta * shifted))) / beta
    else:
        lse = max_ov
    
    sum_penalty = lam_sum * (np.sum(h) - TARGET_SUM) ** 2
    return lse + sum_penalty

def construct():
    """
    Construct the step heights h to minimize the maximum overlap of h with shifted 1-h.
    
    We formulate the problem using linear cross-correlation (overlap) and use a 
    smooth surrogate (Log-Sum-Exp) for the maximum overlap to enable gradient-based 
    refinement via L-BFGS-B. We start from a diverse set of initial candidates 
    (including spectral-inspired sinusoids) and refine them.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    bounds = [(0.0, 1.0) for _ in range(N)]
    rng = np.random.default_rng(123)
    
    # Initial best: uniform
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Generate diverse initial candidates
    candidates = []
    candidates.append(np.full(N, 0.5))
    
    # Alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 0.6
    h_alt[1::2] = 0.4
    candidates.append(h_alt)
    
    # Sinusoidal (Spectral initialization idea)
    t = np.arange(N)
    for freq in range(1, 20):
        for amp in [0.1, 0.2, 0.3, 0.4]:
            for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                trial = 0.5 + amp * np.sin(2*np.pi*freq*t/N + phase)
                candidates.append(trial.copy())
    
    # Random
    for i in range(50):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        candidates.append(h_rand)
    
    # Evaluate all candidates
    for h_cand in candidates:
        h_proj = _project_to_feasible(h_cand)
        score = _max_overlap(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj
    
    # Optimization phases with increasing beta
    beta_schedule = [5, 10, 20, 50, 100, 200]
    
    for beta in beta_schedule:
        if time.time() > deadline - 15.0:
            break
        
        # Try from best solution
        x0 = best_h.copy()
        
        def obj_func(x):
            return _objective_lse(x, beta, 100.0)
        
        try:
            result = minimize(
                obj_func, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
        
        # Try from random restarts
        for _ in range(5):
            if time.time() > deadline - 15.0:
                break
            
            x0 = rng.uniform(0.2, 0.8, size=N)
            x0 = _project_to_feasible(x0)
            
            def obj_rand(x):
                return _objective_lse(x, beta, 100.0)
            
            try:
                result = minimize(
                    obj_rand, x0, method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-7}
                )
                h_new = _project_to_feasible(result.x)
                score_new = _max_overlap(h_new)
                if score_new < best_score:
                    best_score = score_new
                    best_h = h_new
            except Exception:
                pass
    
    # Final polish with high beta
    if time.time() < deadline - 5.0:
        def obj_polish(x):
            return _objective_lse(x, 500.0, 1000.0)
        
        try:
            result = minimize(
                obj_polish, best_h, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_polished = _project_to_feasible(result.x)
            score_polished = _max_overlap(h_polished)
            if score_polished < best_score:
                best_score = score_polished
                best_h = h_polished
        except Exception:
            pass
    
    # Ensure final validity
    best_h = _project_to_feasible(best_h)
    
    # Final precision polish: ensure sum is exactly N/2
    sum_h = np.sum(best_h)
    diff = TARGET_SUM - sum_h
    if abs(diff) > 1e-12:
        # Add to the element with the most room (closest to 0.5?) or just the first one
        # To keep it simple, distribute to all? No, that changes the distribution.
        # Add to index 0, check bounds
        if best_h[0] + diff <= 1.0 and best_h[0] + diff >= 0.0:
            best_h[0] += diff
        else:
            # Fallback: scale
            best_h = best_h * (TARGET_SUM / sum_h)
            best_h = np.clip(best_h, 0.0, 1.0)
            # Re-project
            best_h = _project_to_feasible(best_h)
    
    return best_h

# EVOLVE-BLOCK-END
