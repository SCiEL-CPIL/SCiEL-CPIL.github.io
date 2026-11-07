import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _value(h):
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _soft_value(h, lam):
    corr = np.correlate(h, 1 - h, mode="full")
    # soft-max via log-sum-exp
    m = np.max(corr)
    s = np.sum(np.exp(lam * (corr - m)))
    soft_max = m + np.log(s) / lam
    return soft_max / N * 2

def _grad_soft_value(h, lam):
    # We'll use finite differences for gradient if needed, or analytic
    # Actually, let's use scipy's numerical gradient via finite differences in minimize
    pass

def construct():
    t0 = time.time()
    deadline = 150.0 - 5.0  # safety margin
    
    # Start from seed
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_val = _value(h0)
    
    # Multi-stage optimization
    # Stage 1: low lambda (smooth), Stage 2: medium, Stage 3: high, then polish
    stages = [
        (10.0, 50),   # lambda, max_iter
        (50.0, 100),
        (200.0, 200),
    ]
    
    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2}]
    
    for lam, max_iter in stages:
        if time.time() > deadline:
            break
        
        def obj(x):
            return _soft_value(x, lam)
        
        res = minimize(
            obj, h0, method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': max_iter, 'ftol': 1e-10}
        )
        
        if res.success or res.fun < best_val:
            h_candidate = res.x
            # Clip and fix sum
            h_candidate = np.clip(h_candidate, 0, 1)
            # Adjust to satisfy sum constraint
            diff = N/2 - np.sum(h_candidate)
            h_candidate += diff / N
            h_candidate = np.clip(h_candidate, 0, 1)
            
            v = _value(h_candidate)
            if v < best_val:
                best_h = h_candidate.copy()
                best_val = v
                h0 = h_candidate.copy()
    
    # Final polish with high lambda
    if time.time() < deadline:
        lam = 1000.0
        def obj(x):
            return _soft_value(x, lam)
        
        res = minimize(
            obj, h0, method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        if res.fun < best_val:
            h_candidate = np.clip(res.x, 0, 1)
            diff = N/2 - np.sum(h_candidate)
            h_candidate += diff / N
            h_candidate = np.clip(h_candidate, 0, 1)
            v = _value(h_candidate)
            if v < best_val:
                best_h = h_candidate.copy()
                best_val = v
    
    # Final check and return
    best_h = np.clip(best_h, 0, 1)
    diff = N/2 - np.sum(best_h)
    best_h += diff / N
    best_h = np.clip(best_h, 0, 1)
    
    # Verify sum constraint
    assert abs(np.sum(best_h) - N/2) < 1e-9 * N, "Sum constraint violated"
    assert np.all(np.isfinite(best_h)), "Non-finite values"
    
    return best_h
