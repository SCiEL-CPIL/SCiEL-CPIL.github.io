import time
import numpy as np
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 150.0
SAFETY = 10.0
START_TIME = time.time()

def _value(h):
    """
    Computes the objective value for a given height array h.
    Vectorized for speed.
    """
    h = np.asarray(h, dtype=float)
    # Ensure non-negative and finite
    h = np.clip(h, 0, None)
    if not np.all(np.isfinite(h)):
        return 0.0
    
    g = np.convolve(h, h)
    m = len(g)
    # y = [0, g[0], ..., g[m-1], 0]
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    # Spacing w
    w = 1.0 / (m + 1)
    
    # l2 = sum over i in 0..m of (w / 3) * (y[i]^2 + y[i] * y[i+1] + y[i+1]^2)
    y_curr = y[:-1]
    y_next = y[1:]
    term = y_curr**2 + y_curr * y_next + y_next**2
    l2 = (w / 3.0) * np.sum(term)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
        
    return l2 / (norm_1 * norm_inf)

def _neg_value(h):
    """
    Objective function for minimization.
    Returns negative value.
    """
    return -_value(h)

def _concentrated(beta):
    # h_i = beta ** i, i=0..N-1, concentrated at one end
    return np.array([beta ** i for i in range(N)])

def _cosine_peak(c, amp, width):
    # smooth bump centered at c in [0,1)
    x = np.arange(N) / N
    d = np.minimum(np.abs(x - c), 1.0 - np.abs(x - c))
    return amp * np.cos(np.pi * d / width) ** 2

def _gaussian(center, sigma):
    x = np.arange(N) / N
    return np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

def construct():
    start = time.time()
    deadline = start + TIME_LIMIT - SAFETY
    rng = np.random.default_rng(12345)

    # Initial seed: uniform
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    # Generate diverse initial guesses
    initial_guesses = []
    
    # 1. Uniform (Seed)
    initial_guesses.append(np.ones(N))
    
    # 2. Concentrated at edges (often good for L-inf ratio)
    for beta in [0.9, 0.8, 0.7, 0.5]:
        initial_guesses.append(_concentrated(beta))
        # Also concentrated at the other end
        initial_guesses.append(_concentrated(beta)[::-1])
        
    # 3. Cosine peaks at various positions and widths
    for c in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for width in [0.2, 0.4, 0.6]:
            initial_guesses.append(_cosine_peak(c, 1.0, width))
            
    # 4. Gaussians
    for sigma in [0.05, 0.1, 0.15, 0.2]:
        for center in [0.0, 0.5, 1.0]:
            initial_guesses.append(_gaussian(center, sigma))
            
    # 5. Random non-negative
    for _ in range(5):
        initial_guesses.append(rng.random(N) * 2.0)
        
    # 6. Sparse random (mostly zeros, some spikes)
    for _ in range(5):
        h_rand = np.zeros(N)
        indices = rng.choice(N, size=5, replace=False)
        h_rand[indices] = rng.random(5) + 0.5
        initial_guesses.append(h_rand)

    bounds = [(0, None)] * N
    
    # Optimize each initial guess
    for h0 in initial_guesses:
        if time.time() > deadline:
            break
            
        h0 = np.maximum(h0, 0)
        if np.sum(h0) == 0:
            continue
            
        try:
            # Use L-BFGS-B which is efficient for bounded problems
            # maxiter set high enough to converge, but we check time
            res = minimize(
                _neg_value, 
                h0, 
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 500,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                    'maxls': 50
                }
            )
            
            if res.x is not None:
                h_opt = np.clip(res.x, 0, None)
                val_opt = _value(h_opt)
                
                if val_opt > best_v:
                    best_v = val_opt
                    best_h = h_opt
                    
        except Exception:
            pass
            
    # If we have time, do a final polish on the best solution found
    # using a different optimizer or tighter tolerance, just in case
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _neg_value, 
                best_h, 
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'maxls': 100
                }
            )
            if res.x is not None:
                h_polished = np.clip(res.x, 0, None)
                val_polished = _value(h_polished)
                if val_polished > best_v:
                    best_v = val_polished
                    best_h = h_polished
        except Exception:
            pass

    # Final safety: ensure finite and non-negative
    best_h = np.maximum(best_h, 0)
    if not np.all(np.isfinite(best_h)):
        best_h = np.ones(N)
        
    return [float(x) for x in best_h]
