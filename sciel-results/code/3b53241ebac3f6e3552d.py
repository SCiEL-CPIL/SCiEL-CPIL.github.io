import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_value(h):
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return -1.0
    h = np.maximum(h, 0.0)
    g = np.convolve(h, h)
    m = len(g)  # 99
    y = np.concatenate(([0.0], g, [0.0]))  # length 101
    w = 1.0 / 100.0
    # l2 = sum_{i=0}^{m} (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # i from 0 to m (99), so y[0..99] and y[1..100]
    yi = y[:m+1]
    yip1 = y[1:m+2]
    terms = yi**2 + yi*yip1 + yip1**2
    l2 = (w/3.0) * np.sum(terms)
    norm_1 = np.sum(g) / (m + 1)  # g >= 0
    norm_inf = np.max(g)
    if norm_1 <= 0 or norm_inf <= 0:
        return -1.0
    return l2 / (norm_1 * norm_inf)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_h = [1.0] * N
    best_val = compute_value(best_h)
    
    def obj_neg(x):
        h = np.maximum(x, 0.0)
        s = np.sum(h)
        if s <= 1e-12:
            return 1e10
        # Normalize to fix scale (since scale-invariant)
        h_norm = h / s * N  # keep sum = N for numerical stability
        return -compute_value(h_norm)
    
    np.random.seed(123)
    
    # Initializations
    inits = []
    inits.append(np.ones(N))
    inits.append(np.exp(-((np.arange(N) - N/2.0)**2) / (2*8.0**2)))
    inits.append(np.exp(-((np.arange(N) - N/2.0)**2) / (2*15.0**2)))
    inits.append(np.exp(-((np.arange(N) - N/2.0)**2) / (2*25.0**2)))
    inits.append(np.sin(np.pi * np.arange(N) / N)**2 + 0.05)
    inits.append(np.ones(N) * 0.2 + np.random.rand(N) * 0.8)
    inits.append(np.ones(N))
    inits[-1][N//2-5:N//2+5] = 3.0
    inits.append(np.ones(N))
    inits[-1][:15] = 0.1
    inits[-1][-15:] = 0.1
    inits.append(np.ones(N))
    inits[-1][N//2-3:N//2+3] = 5.0
    inits.append(np.exp(-np.abs(np.arange(N) - N/2.0) / 3.0))
    inits.append(np.exp(-np.abs(np.arange(N) - N/2.0) / 10.0))
    inits.append(np.random.rand(N))
    inits.append(np.ones(N) * 0.5 + 0.5 * np.cos(2*np.pi*np.arange(N)/N))
    inits.append(np.ones(N) * 0.5 + 0.5 * np.cos(4*np.pi*np.arange(N)/N))
    inits.append(np.ones(N) * 0.5 + 0.5 * np.sin(2*np.pi*np.arange(N)/N))
    
    for idx, h0 in enumerate(inits):
        if time.time() > deadline - 20:
            break
        h0 = np.maximum(h0, 0.0)
        if np.sum(h0) < 1e-12:
            continue
        try:
            res = minimize(obj_neg, h0, method='L-BFGS-B',
                           bounds=[(0.0, None)] * N,
                           options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-8})
            h_opt = np.maximum(res.x, 0.0)
            if np.sum(h_opt) > 1e-12:
                val = compute_value(h_opt)
                if val > best_val:
                    best_val = val
                    best_h = h_opt.tolist()
        except Exception:
            pass
    
    # Also try a grid search over a few parametric families
    # Gaussian family: h[i] = exp(-((i-c)^2)/(2*s^2))
    for c in [N/2.0, N/2.0 - 5, N/2.0 + 5]:
        if time.time() > deadline - 10:
            break
        for s in [3, 5, 7, 10, 15, 20, 25, 30]:
            h_g = np.exp(-((np.arange(N) - c)**2) / (2*s*s))
            val = compute_value(h_g)
            if val > best_val:
                best_val = val
                best_h = h_g.tolist()
    
    return best_h
