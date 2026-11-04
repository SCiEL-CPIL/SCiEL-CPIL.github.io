import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _compute_linear_overlaps(h):
    """Compute linear cross-correlation of h with (1-h) for all non-zero shifts."""
    g = 1.0 - h
    c = np.correlate(h, g, mode='full')
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
    
    lo, hi = -1.0, 1.0
    
    for _ in range(80):
        mid = (lo + hi) / 2
        s = np.sum(np.clip(h - mid, 0, 1))
        if abs(s - target) < 1e-14:
            break
        if s > target:
            lo = mid
        else:
            hi = mid
    
    lam = (lo + hi) / 2
    h = np.clip(h - lam, 0, 1)
    
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 1e-10) & (h < 1 - 1e-10)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0, 1)
    
    return h

def _objective_lse(h, beta):
    """LSE-smoothed max overlap."""
    overlaps = _compute_linear_overlaps(h)
    if len(overlaps) == 0:
        return 0.0
    max_ov = np.max(overlaps)
    if beta > 0:
        shifted = overlaps - max_ov
        shifted = np.clip(shifted, -50, 50)
        lse = max_ov + np.log(np.sum(np.exp(beta * shifted))) / beta
        return lse
    return max_ov

def _objective_epigraph(x, beta, lam_sum):
    """Epigraph formulation: minimize t + penalty, s.t. overlaps <= t."""
    h = np.clip(x[:N], 0, 1)
    t = x[N]
    
    overlaps = _compute_linear_overlaps(h)
    if len(overlaps) == 0:
        max_ov = 0.0
    else:
        max_ov = np.max(overlaps)
    
    # Soft constraint: penalize violations of overlaps <= t
    violations = np.maximum(0, overlaps - t)
    soft_pen = beta * np.sum(violations ** 2)
    
    sum_penalty = lam_sum * (np.sum(h) - N / 2.0) ** 2
    return t + soft_pen + sum_penalty

def _spectral_init(rng):
    """Generate diverse initial candidates."""
    candidates = []
    t = np.arange(N)
    
    candidates.append(np.full(N, 0.5))
    
    for base in [0.3, 0.4, 0.45, 0.5]:
        h_alt = np.zeros(N)
        h_alt[::2] = base + 0.1
        h_alt[1::2] = base - 0.1
        candidates.append(h_alt.copy())
    
    for n_harm in range(1, 6):
        for total_amp in [0.25, 0.35, 0.45]:
            amps = rng.dirichlet(np.ones(n_harm)) * total_amp
            phases = rng.uniform(0, 2*np.pi, size=n_harm)
            h = np.full(N, 0.5)
            for k in range(n_harm):
                freq = (k + 1)
                h += amps[k] * np.sin(2*np.pi*freq*t/N + phases[k])
            candidates.append(h.copy())
    
    for freq in range(1, 15):
        for amp in [0.2, 0.3, 0.4]:
            for phase in [0, np.pi/3, 2*np.pi/3]:
                h = 0.5 + amp * np.sin(2*np.pi*freq*t/N + phase)
                candidates.append(h.copy())
    
    for _ in range(30):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        candidates.append(h_rand)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    
    bounds_h = [(0.0, 1.0) for _ in range(N)]
    bounds_full = bounds_h + [(-1.0, 1.0)]  # t can be negative
    rng = np.random.default_rng(42)
    
    candidates = _spectral_init(rng)
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    for h_cand in candidates:
        h_proj = _project_to_feasible(h_cand)
        score = _max_overlap(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj
    
    # Phase 1: LSE optimization with increasing beta
    beta_schedule = [5, 15, 40, 100, 250]
    
    for beta in beta_schedule:
        if time.time() > deadline - 20.0:
            break
        
        x0 = best_h.copy()
        
        def obj_lse(x):
            h = np.clip(x, 0, 1)
            return _objective_lse(h, beta) + 500.0 * (np.sum(h) - N/2.0)**2
        
        try:
            result = minimize(
                obj_lse, x0, method='L-BFGS-B', bounds=bounds_h,
                options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_new = _project_to_feasible(result.x)
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
        
        num_restarts = 4 if beta <= 40 else 2
        for _ in range(num_restarts):
            if time.time() > deadline - 20.0:
                break
            x0 = _project_to_feasible(rng.uniform(0.1, 0.9, size=N))
            try:
                result = minimize(
                    obj_lse, x0, method='L-BFGS-B', bounds=bounds_h,
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                h_new = _project_to_feasible(result.x)
                score_new = _max_overlap(h_new)
                if score_new < best_score:
                    best_score = score_new
                    best_h = h_new
            except Exception:
                pass
    
    # Phase 2: Epigraph formulation for sharper max handling
    if time.time() < deadline - 15.0:
        t0 = best_score + 0.01
        x0_ep = np.concatenate([best_h, [t0]])
        
        def obj_ep(x):
            return _objective_epigraph(x, 50.0, 500.0)
        
        try:
            result = minimize(
                obj_ep, x0_ep, method='L-BFGS-B', bounds=bounds_full,
                options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_new = _project_to_feasible(result.x[:N])
            score_new = _max_overlap(h_new)
            if score_new < best_score:
                best_score = score_new
                best_h = h_new
        except Exception:
            pass
    
    # Phase 3: Efficient local polish via gradient-based steps
    if time.time() < deadline - 5.0:
        # Use a few more LSE passes with very high beta for final sharpening
        for beta_final in [500, 1000, 2000]:
            if time.time() > deadline - 5.0:
                break
            x0 = best_h.copy()
            def obj_final(x):
                h = np.clip(x, 0, 1)
                return _objective_lse(h, beta_final) + 5000.0 * (np.sum(h) - N/2.0)**2
            try:
                result = minimize(
                    obj_final, x0, method='L-BFGS-B', bounds=bounds_h,
                    options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                h_polished = _project_to_feasible(result.x)
                score_polished = _max_overlap(h_polished)
                if score_polished < best_score:
                    best_score = score_polished
                    best_h = h_polished
            except Exception:
                pass
    
    best_h = _project_to_feasible(best_h)
    return best_h
