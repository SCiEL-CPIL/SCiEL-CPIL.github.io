import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 50

def _autoconv_max(h):
    n = len(h)
    ac_max = np.sum(h * h)  # lag 0
    for k in range(1, n):
        val = np.sum(h[:n-k] * h[k:])
        if val > ac_max:
            ac_max = val
    return ac_max

def _ratio_from_log(x):
    h = np.exp(x)
    s = np.sum(h)
    if s <= 0:
        return -np.inf
    h = h / s  # normalize, scale-invariant
    l2sq = np.sum(h * h)
    l1 = np.sum(h)  # = 1 after normalization
    acmax = _autoconv_max(h)
    if acmax <= 0:
        return -np.inf
    return l2sq / (l1 * acmax)

def _neg_ratio_from_log(x):
    return -_ratio_from_log(x)

def construct():
    t0 = time.time()
    deadline = t0 + 130.0
    
    best_score = 0.0
    best_x = np.zeros(N)
    best_score = _ratio_from_log(best_x)
    
    bounds = [(-10.0, 10.0)] * N
    
    elapsed = time.time() - t0
    if elapsed < deadline - 50.0:
        try:
            result = differential_evolution(
                _neg_ratio_from_log, bounds,
                seed=42, maxiter=150, popsize=12,
                tol=1e-5, polish=True,
                update_interval=5
            )
            test_score = _ratio_from_log(result.x)
            if test_score > best_score:
                best_score = test_score
                best_x = result.x.copy()
        except Exception:
            pass
    
    elapsed = time.time() - t0
    if elapsed < deadline - 5.0:
        try:
            res = minimize(
                _neg_ratio_from_log, best_x,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-14}
            )
            test_score = _ratio_from_log(res.x)
            if test_score > best_score:
                best_score = test_score
                best_x = res.x.copy()
        except Exception:
            pass
    
    h = np.exp(best_x)
    return [float(v) for v in h]
