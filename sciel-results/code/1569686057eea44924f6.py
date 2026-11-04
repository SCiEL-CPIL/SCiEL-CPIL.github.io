import numpy as np
import time

N = 50
W = 1.0 / N  # width of each interval

def _autoconv_sup(h):
    """Compute supremum of autoconvolution of step function with heights h."""
    # Autoconvolution at lag 0 is w * sum(h_i^2)
    # For nonnegative step functions, this is the maximum
    return W * np.sum(h**2)

def _ratio(h):
    """Compute the ratio: L2^2 / (L1 * sup_autoconv)."""
    l2_sq = W * np.sum(h**2)
    l1 = W * np.sum(h)
    sup_ac = _autoconv_sup(h)
    if l1 <= 0 or sup_ac <= 0:
        return 0.0
    return l2_sq / (l1 * sup_ac)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # Try various structured initializations
    inits = [
        np.ones(N),
        np.linspace(0.1, 1.9, N),
        np.array([1.0 if i % 2 == 0 else 0.1 for i in range(N)]),
        np.array([1.0 if i % 2 == 1 else 0.1 for i in range(N)]),
        np.exp(-np.linspace(0, 3, N)),
        np.exp(-np.linspace(3, 0, N)),
        np.concatenate([np.ones(N//2), np.zeros(N - N//2)]),
        np.concatenate([np.zeros(N//2), np.ones(N - N//2)]),
        np.abs(np.sin(np.linspace(0, 2*np.pi, N))) + 0.1,
        np.array([1.0 - 0.8*np.cos(2*np.pi*i/N) for i in range(N)]),
    ]
    
    for h0 in inits:
        if time.time() > deadline:
            break
        h0 = np.maximum(h0, 1e-10)
        s = _ratio(h0)
        if s > best_score:
            best_score = s
            best_h = h0.copy()
    
    # Use scipy.optimize to refine
    try:
        from scipy.optimize import minimize
        for h0 in inits[:5]:
            if time.time() > deadline:
                break
            h0 = np.maximum(h0, 1e-10)
            
            def neg_ratio(x):
                x = np.maximum(x, 1e-10)
                return -_ratio(x)
            
            res = minimize(neg_ratio, h0, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            if time.time() > deadline:
                break
            h_opt = np.maximum(res.x, 1e-10)
            s = _ratio(h_opt)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()
    except Exception:
        pass
    
    return list(best_h)
