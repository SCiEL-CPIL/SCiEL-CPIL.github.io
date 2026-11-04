import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _compute_overlaps(h):
    """Compute linear overlap of h with shifted (1-h) for all shifts."""
    g = 1.0 - h
    c = np.correlate(h, g, mode='full')
    # Exclude the zero shift (center)
    c1 = c[:N]      # negative shifts
    c2 = c[N+1:]    # positive shifts
    return np.concatenate([c1, c2])

def _max_overlap(h):
    """Return the maximum overlap value."""
    h = np.clip(h, 0, 1)
    overlaps = _compute_overlaps(h)
    return np.max(overlaps)

def _project_to_feasible(h):
    """Project h onto [0,1]^N with sum = N/2 using bisection on water-filling."""
    h = np.clip(h, 0, 1)
    target = N / 2.0
    current = np.sum(h)
    if abs(current - target) < 1e-12:
        return h
    
    # Water-filling: find lambda such that sum(clip(h + lambda, 0, 1)) = target
    # If current > target, we need to decrease; if current < target, increase
    if current > target:
        # Need to reduce: h_new = clip(h - delta, 0, 1) for some delta > 0
        lo, hi = 0.0, 1.0
    else:
        # Need to increase: h_new = clip(h + delta, 0, 1) for some delta > 0
        lo, hi = 0.0, 1.0
    
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if current > target:
            h_trial = np.clip(h - mid, 0, 1)
        else:
            h_trial = np.clip(h + mid, 0, 1)
        s = np.sum(h_trial)
        if current > target:
            if s > target:
                lo = mid
            else:
                hi = mid
        else:
            if s < target:
                lo = mid
            else:
                hi = mid
        if hi - lo < 1e-14:
            break
    
    mid = (lo + hi) / 2.0
    if current > target:
        h = np.clip(h - mid, 0, 1)
    else:
        h = np.clip(h + mid, 0, 1)
    
    # Fine-tune to get exact sum
    diff = target - np.sum(h)
    if abs(diff) > 1e-13:
        if diff > 0:
            room = 1.0 - h
            room[h >= 1.0 - 1e-15] = 0
            total_room = room.sum()
            if total_room > 1e-15:
                h += room * (diff / total_room)
                h = np.clip(h, 0, 1)
        else:
            avail = h
            avail[h <= 1e-15] = 0
            total_avail = avail.sum()
            if total_avail > 1e-15:
                h -= avail * (-diff / total_avail)
                h = np.clip(h, 0, 1)
    return h

def _objective_lse(x, beta, lam_sum):
    """Objective: minimize max overlap using LogSumExp approximation."""
    h = np.clip(x, 0, 1)
    overlaps = _compute_overlaps(h)
    
    max_ov = np.max(overlaps)
    lse = max_ov + np.log(np.sum(np.exp(beta * (overlaps - max_ov)))) / beta
    
    sum_penalty = lam_sum * (np.sum(h) - N / 2.0) ** 2
    return lse + sum_penalty

def construct():
    """
    Returns the optimal sequence of step heights.
    
    Minimize the maximum overlap of h with shifted 1-h.
    Uses LSE approximation with increasing beta for convergence to max.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    bounds_h = [(0.0, 1.0) for _ in range(N)]
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Generate diverse initial candidates
    rng = np.random.default_rng(42)
    candidates = []
    
    # Uniform
    candidates.append(np.full(N, 0.5))
    
    # Alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 0.6
    h_alt[1::2] = 0.4
    candidates.append(h_alt)
    
    # Sinusoidal modulations
    t_idx = np.arange(N)
    for freq in range(1, 15):
        for amp in [0.1, 0.2, 0.3, 0.4, 0.45]:
            for phase in [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]:
                trial = 0.5 + amp * np.sin(2*np.pi*freq*t_idx/N + phase)
                candidates.append(trial.copy())
    
    # Random restarts
    for i in range(30):
        h_rand = rng.uniform(0.05, 0.95, size=N)
        candidates.append(h_rand.copy())
    
    # Evaluate all candidates first
    for h_cand in candidates:
        h_proj = _project_to_feasible(h_cand)
        score = _max_overlap(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj
    
    # LSE optimization with increasing beta
    beta_schedule = [5, 10, 20, 40, 80, 150, 300]
    
    for beta in beta_schedule:
        if time.time() > deadline - 20.0:
            break
        
        x0 = best_h.copy()
        
        def obj_func(x, b=beta):
            return _objective_lse(x, b, 500.0)
        
        try:
            result = minimize(
                obj_func,
                x0,
                method='L-BFGS-B',
                bounds=bounds_h,
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
    
    # Final high-precision polish with very high beta
    if time.time() < deadline - 10.0:
        for beta in [500, 1000]:
            if time.time() > deadline - 5.0:
                break
            x0 = best_h.copy()
            
            def obj_polish(x, b=beta):
                return _objective_lse(x, b, 1000.0)
            
            try:
                result = minimize(
                    obj_polish,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds_h,
                    options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-11}
                )
                
                h_new = _project_to_feasible(result.x)
                score_new = _max_overlap(h_new)
                if score_new < best_score:
                    best_score = score_new
                    best_h = h_new
            except Exception:
                pass
    
    return best_h
