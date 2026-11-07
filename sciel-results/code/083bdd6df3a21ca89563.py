import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _correlation_value(h):
    """Compute the evaluator's value: max(corr) / N * 2"""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _project_to_feasible(h):
    """Project h to [0,1] and sum=N/2 using bisection on a global shift"""
    target = N / 2.0
    
    # We want to find s such that sum(clip(h + s, 0, 1)) = target
    # f(s) = sum(clip(h + s, 0, 1)) is monotonic non-decreasing.
    # Lower bound: s such that all h_i + s <= 0 -> s <= -max(h)
    # Upper bound: s such that all h_i + s >= 1 -> s >= 1 - min(h)
    lo = -np.max(h) - 1.0
    hi = 1.0 - np.min(h) + 1.0
    
    # Bisection
    for _ in range(50):
        mid = (lo + hi) / 2.0
        val = np.sum(np.clip(h + mid, 0.0, 1.0)) - target
        if val > 0:
            hi = mid
        else:
            lo = mid
            
    s = (lo + hi) / 2.0
    h_proj = np.clip(h + s, 0.0, 1.0)
    
    # Final check for floating point precision
    diff = np.sum(h_proj) - target
    if abs(diff) > 1e-12:
        # Adjust one element slightly if needed, though bisection should be very accurate
        # This is a fallback
        if diff > 0:
            idx = np.argmax(h_proj)
        else:
            idx = np.argmin(h_proj)
        h_proj[idx] -= diff
        h_proj = np.clip(h_proj, 0.0, 1.0)
        
    return h_proj

def _soft_max(corr, lam):
    """Soft max using log-sum-exp for numerical stability"""
    m = np.max(corr)
    if m == 0:
        return 0.0
    # log(sum(exp(corr - m))) + m
    # = log(sum(exp(corr - m))) + m
    # To avoid overflow in exp, we shift by m.
    # exp((corr - m) * lam) might still overflow if lam is large and corr-m is large?
    # No, corr - m <= 0, so exp is in (0, 1].
    # However, if lam is large, (corr - m) * lam can be very negative, underflowing to 0.
    # This is fine, sum will be dominated by the max term (which is 1).
    # The term log(sum(exp(...))) will be close to log(1 + small) ~ small.
    # So result is m + small.
    
    # Actually, for very large lam, the soft max approaches the hard max.
    # We want to minimize the max.
    # Using soft max as a smooth approximation.
    
    # Alternative: just use the max if we are in the final polish phase.
    # But for optimization, we need gradients.
    
    # Let's use a simpler soft max: (sum(exp(lam * corr))) ** (1/lam) is not stable.
    # Log-sum-exp is better: m + (1/lam) * log(sum(exp(lam * (corr - m))))
    
    shifted = lam * (corr - m)
    # Clip shifted to avoid underflow issues? No, underflow to 0 is fine.
    # Overflow? shifted <= 0, so exp(shifted) <= 1. No overflow.
    
    s = np.sum(np.exp(shifted))
    if s == 0:
        return m
    return m + np.log(s) / lam

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial best: Seed
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    # We will use a smooth surrogate objective.
    # Objective: soft_max(correlate(h, 1-h))
    # Constraints: 0 <= h <= 1, sum(h) = N/2.
    # We handle constraints via projection inside the objective function.
    
    # Phase 1: Global-ish search with L-BFGS-B from multiple starts?
    # Or just one strong start and refinement.
    # The seed is 0.5 everywhere. 
    # Let's try to create some structure.
    # A known good structure for minimizing max correlation might be alternating or sinusoidal.
    # Let's try a few different initializations.
    
    inits = [
        np.full(N, 0.5),
        np.linspace(0, 1, N),
        np.linspace(1, 0, N),
        np.sin(np.linspace(0, 2*np.pi, N)) * 0.5 + 0.5,
        np.cos(np.linspace(0, 2*np.pi, N)) * 0.5 + 0.5,
        np.sign(np.sin(np.linspace(0, 2*np.pi, N))) * 0.5 + 0.5,
        np.random.RandomState(42).uniform(0, 1, N),
        np.random.RandomState(123).uniform(0, 1, N),
    ]
    
    # Project all inits
    inits = [_project_to_feasible(x) for x in inits]
    
    # Optimization loop
    # We will run L-BFGS-B with increasing lambda (sharper max approximation)
    # and multiple restarts if time permits.
    
    # Define the objective function factory
    def make_obj(lam):
        def obj(x):
            if time.time() > deadline - 5:
                return 1e9
            h = _project_to_feasible(x)
            corr = np.correlate(h, 1 - h, mode="full")
            return _soft_max(corr, lam)
        return obj
    
    bounds = [(0.0, 1.0)] * N
    
    # Start with moderate lambda
    # Lambda too small -> too smooth, might miss the max.
    # Lambda too large -> nearly non-smooth, L-BFGS-B might struggle.
    # Let's ramp up lambda.
    
    current_h = best_h.copy()
    
    for lam in [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]:
        if time.time() > deadline - 15:
            break
        
        obj = make_obj(lam)
        
        # Try optimizing from current best
        try:
            res = minimize(
                obj,
                current_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 50}
            )
            h_opt = _project_to_feasible(res.x)
            val_opt = _correlation_value(h_opt)
            
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
                current_h = best_h.copy()
            
        except Exception:
            pass
        
        # If time permits, try another random start with current lambda
        if time.time() < deadline - 30:
            # Pick a random init
            init_idx = int(time.time() * 1000) % len(inits)
            x0 = inits[init_idx].copy()
            # Add some noise
            x0 += np.random.RandomState(init_idx).normal(0, 0.1, N)
            x0 = _project_to_feasible(x0)
            
            try:
                res = minimize(
                    obj,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-8, 'maxls': 50}
                )
                h_opt = _project_to_feasible(res.x)
                val_opt = _correlation_value(h_opt)
                
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
                    current_h = best_h.copy()
                    
            except Exception:
                pass
    
    # Final polish: use a very high lambda soft max for tight convergence
    if time.time() < deadline - 5:
        lam_final = 10000.0
        obj_final = make_obj(lam_final)
        try:
            res = minimize(
                obj_final,
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100}
            )
            h_opt = _project_to_feasible(res.x)
            val_opt = _correlation_value(h_opt)
            
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass
            
    # One more polish with even higher lambda if time permits
    if time.time() < deadline - 2:
        lam_final2 = 20000.0
        obj_final2 = make_obj(lam_final2)
        try:
            res = minimize(
                obj_final2,
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100}
            )
            h_opt = _project_to_feasible(res.x)
            val_opt = _correlation_value(h_opt)
            
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Ensure final result is strictly feasible
    best_h = _project_to_feasible(best_h)
    
    return best_h
