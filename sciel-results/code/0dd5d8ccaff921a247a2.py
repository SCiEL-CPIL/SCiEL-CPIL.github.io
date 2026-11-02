import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _score(h):
    h = np.asarray(h, dtype=float)
    l2_sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 == 0 or l2_sq == 0:
        return 0.0
    # Autoconvolution: (h*h)[k] = sum_i h[i] * h[i+k]
    # For nonnegative h, max is at lag 0 = l2_sq
    conv_max = l2_sq
    return l2_sq / (l1 * conv_max)

def _neg_score(h):
    return -_score(h)

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin
    
    best_val = 0.0
    best_h = np.ones(N)
    
    # Update with seed
    v = _score(best_h)
    if v > best_val:
        best_val = v
        best_h = best_h.copy()
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initial points
    starts = []
    
    # Uniform
    starts.append(np.ones(N))
    
    # Spikes at various positions
    for pos in [0, 10, 25, 39, 49]:
        s = np.zeros(N)
        s[pos] = 1.0
        starts.append(s)
    
    # Two spikes
    for pos1, pos2 in [(0, 49), (10, 39), (0, 25)]:
        s = np.zeros(N)
        s[pos1] = 1.0
        s[pos2] = 1.0
        starts.append(s)
    
    # Gaussian-like
    x = np.linspace(-1, 1, N)
    starts.append(np.exp(-x**2 / 0.1))
    starts.append(np.exp(-x**2 / 0.5))
    
    # Bimodal
    x = np.linspace(-1, 1, N)
    starts.append(np.exp(-(x-0.5)**2/0.05) + np.exp(-(x+0.5)**2/0.05))
    
    # Random
    for _ in range(20):
        s = rng.random(N)
        starts.append(s)
    
    # Random with sparsity
    for _ in range(10):
        s = rng.random(N)
        mask = rng.random(N) > 0.7
        s[mask] = 0.0
        starts.append(s)
    
    for i, x0 in enumerate(starts):
        if time.time() > deadline:
            break
        
        x0 = np.maximum(x0, 0.0)
        if np.sum(x0) == 0:
            continue
        
        # Try BFGS
        try:
            res = minimize(_neg_score, x0, method='BFGS', 
                          bounds=[(0, None)] * N,
                          options={'maxiter': 200, 'gtol': 1e-8})
            h_opt = np.maximum(res.x, 0.0)
            v = _score(h_opt)
            if v > best_val:
                best_val = v
                best_h = h_opt.copy()
        except Exception:
            pass
        
        # Try Nelder-Mead
        if time.time() > deadline:
            break
        try:
            res = minimize(_neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            h_opt = np.maximum(res.x, 0.0)
            v = _score(h_opt)
            if v > best_val:
                best_val = v
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final refinement with more iterations on best
    if time.time() < deadline - 5:
        try:
            res = minimize(_neg_score, best_h, method='BFGS',
                          bounds=[(0, None)] * N,
                          options={'maxiter': 1000, 'gtol': 1e-10})
            h_opt = np.maximum(res.x, 0.0)
            v = _score(h_opt)
            if v > best_val:
                best_val = v
                best_h = h_opt.copy()
        except Exception:
            pass
    
    return list(best_h)
