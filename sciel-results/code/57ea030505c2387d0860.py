import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # The overlap at shift t is: sum_i h[i] * (1 - h[(i+t)%N]) = N/2 - autocorr_h[t]
    # where autocorr_h[t] = sum_i h[i] * h[(i+t)%N]
    # We want to minimize max_t overlap = N/2 - min_t autocorr_h[t]
    # So we maximize min_t autocorr_h[t]
    
    def compute_autocorr(h):
        """Compute circular autocorrelation of h."""
        return np.array([np.sum(h * np.roll(h, t)) for t in range(N)])
    
    def final_objective(h):
        h_clipped = np.clip(h, 0, 1)
        autocorr = compute_autocorr(h_clipped)
        # Maximize min(autocorr) using smooth min
        beta = 50.0
        m = np.min(autocorr)
        smooth_min = m - np.log(np.sum(np.exp(-beta * (autocorr - m)))) / beta
        # Penalty for sum constraint
        sum_penalty = 1000.0 * (np.sum(h_clipped) - N / 2.0) ** 2
        return -smooth_min + sum_penalty
    
    # Phase 1: Bisection to find a good feasible region
    def feasibility_check(c):
        """Check if there exists h with autocorr_h[t] >= c for all t, 0<=h<=1, sum h=N/2."""
        def obj(h):
            h_clipped = np.clip(h, 0, 1)
            autocorr = compute_autocorr(h_clipped)
            # Violation of autocorr >= c
            viol = np.max(np.maximum(0, c - autocorr))
            # Sum constraint
            sum_viol = np.abs(np.sum(h_clipped) - N / 2.0)
            # Bound violations (should be 0 after clip)
            return viol + 100.0 * sum_viol
        
        x0 = np.full(N, 0.5)
        try:
            res = minimize(obj, x0, method='L-BFGS-B', bounds=[(0,1)]*N,
                          options={'maxiter': 100, 'ftol': 1e-8})
            h = np.clip(res.x, 0, 1)
            autocorr = compute_autocorr(h)
            feasible = np.all(autocorr >= c - 1e-5) and abs(np.sum(h) - N/2.0) < 1e-5
            return feasible, h
        except:
            return False, np.full(N, 0.5)
    
    # Bisection on c = min autocorr
    lo, hi = 0.0, N / 2.0
    best_h = np.full(N, 0.5)
    
    for _ in range(15):
        mid = (lo + hi) / 2.0
        feasible, h = feasibility_check(mid)
        if feasible:
            lo = mid
            best_h = h
        else:
            hi = mid
    
    # Phase 2: Fine-tune from best_h
    x0 = best_h.copy()
    try:
        res = minimize(final_objective, x0, method='L-BFGS-B', 
                      bounds=[(0,1)]*N,
                      options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15})
        h = np.clip(res.x, 0, 1)
    except Exception:
        h = best_h.copy()
    
    # Ensure sum is exactly N/2
    s = np.sum(h)
    if s > 0:
        h = h * (N / 2.0) / s
    h = np.clip(h, 0, 1)
    
    # Fine-tune sum to be exactly N/2
    diff = N / 2.0 - np.sum(h)
    idx = np.argsort(h)[::-1]
    for i in idx:
        if diff > 0:
            add = min(diff, 1.0 - h[i])
            h[i] += add
            diff -= add
        elif diff < 0:
            sub = min(-diff, h[i])
            h[i] -= sub
            diff += sub
        else:
            break
    
    return h
