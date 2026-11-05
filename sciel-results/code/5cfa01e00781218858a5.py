import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 400

def _value(h):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.ones(N)
    best_val = _value(best_h)
    
    def try_update(h_candidate):
        nonlocal best_h, best_val
        if not np.all(np.isfinite(h_candidate)):
            return
        v = _value(h_candidate)
        if v < best_val:
            best_val = v
            best_h = h_candidate.copy()
            
    # Parameterize h using Fourier coefficients to ensure smoothness
    # h_j = a_0 + sum_{k=1}^{K} (a_k cos(2 pi k j / N) + b_k sin(2 pi k j / N))
    # We optimize a_0, a_k, b_k.
    # To keep S non-zero, we fix a_0 to be large enough or optimize it.
    
    K = 10 # Number of harmonics
    params_size = 1 + 2 * K
    
    def params_to_h(params):
        a0 = params[0]
        h = np.full(N, a0)
        for k in range(1, K + 1):
            ak = params[1 + 2*(k-1)]
            bk = params[2 + 2*(k-1)]
            if k > 0:
                angles = 2 * np.pi * k * np.arange(N) / N
                h += ak * np.cos(angles) + bk * np.sin(angles)
        return h

    def objective(params):
        if time.time() > deadline - 10:
            return 1e10
        h = params_to_h(params)
        return _value(h)

    # Initial bounds for DE
    # a0 should be around 1
    # ak, bk should be small
    bounds = [(-5, 5)] + [(-2, 2)] * (2 * K)
    
    # Run Differential Evolution
    try:
        res = differential_evolution(objective, bounds, maxiter=50, popsize=15, 
                                      seed=42, tol=1e-4, polish=False, 
                                      init='sobol', 
                                      update_interval=1)
        try_update(params_to_h(res.x))
    except Exception:
        pass

    # Polish with L-BFGS-B on the full vector h
    if time.time() < deadline - 20:
        def obj_full(h):
            return _value(h)
        
        try:
            res2 = minimize(obj_full, best_h, method='L-BFGS-B', 
                            options={'maxiter': 100, 'gtol': 1e-6})
            try_update(res2.x)
        except Exception:
            pass

    return best_h if np.all(np.isfinite(best_h)) else np.ones(N)
