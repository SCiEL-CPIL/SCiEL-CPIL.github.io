import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _evaluate(h):
    """Compute the task value for given heights h (length N)."""
    h = np.asarray(h, dtype=np.float64)
    if len(h) != N:
        return 0.0
    if not np.all(np.isfinite(h)):
        return 0.0
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    
    g = np.convolve(h, h, mode='full')
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    # Spacing w
    x = np.linspace(-0.5, 0.5, m + 2)
    w = x[1] - x[0]
    
    # l2 computation
    # sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # Note: y has length m+2, indices 0..m+1
    # The sum is over i in 0..m, so terms use y[i], y[i+1] for i=0..m
    # This covers y[0]..y[m+1]
    y_i = y[:m+1]
    y_ip1 = y[1:m+2]
    l2 = (w / 3.0) * np.sum(y_i**2 + y_i * y_ip1 + y_ip1**2)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    
    value = l2 / (norm_1 * norm_inf)
    return value

def _neg_value(h):
    """Negative value for minimization."""
    v = _evaluate(h)
    return -v

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    # Generate diverse initial points
    starts = []
    
    # 1. Uniform
    starts.append(np.ones(N))
    
    # 2. Power-law profiles
    for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]:
        x = np.linspace(-1, 1, N)
        h = np.abs(x) ** (-alpha) if alpha > 0 else np.ones(N)
        h = h / np.max(h)
        starts.append(h)
    
    # 3. Gaussian-like
    x = np.linspace(-1, 1, N)
    for sigma in [0.3, 0.5, 0.7, 1.0]:
        h = np.exp(-x**2 / (2 * sigma**2))
        h = h / np.max(h)
        starts.append(h)
    
    # 4. Sparse spikes
    for spike_pos in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        idx = int(spike_pos * N)
        h = np.zeros(N)
        h[idx] = 1.0
        if idx > 0:
            h[idx-1] = 0.5
        if idx < N-1:
            h[idx+1] = 0.5
        starts.append(h)
    
    # 5. Symmetric bumps
    for center in [0.2, 0.3, 0.5, 0.7, 0.8]:
        x = np.linspace(-1, 1, N)
        width = 0.3
        h = np.exp(-((x - (center*2-1))**2) / (2*width**2))
        h = h / np.max(h)
        starts.append(h)
    
    # 6. Random smooth
    np.random.seed(42)
    for _ in range(10):
        h = np.random.rand(N)
        # Smooth it
        kernel = np.ones(5) / 5
        h = np.convolve(h, kernel, mode='same')
        h = h / np.max(h)
        starts.append(h)
    
    # Evaluate all starts and keep the best ones
    start_vals = []
    for h in starts:
        if time.time() > deadline - 30:
            break
        v = _evaluate(h)
        start_vals.append((v, h.copy()))
    
    # Sort by value descending
    start_vals.sort(key=lambda x: -x[0])
    
    # Take top starts for local optimization
    top_starts = [h for _, h in start_vals[:15]]
    
    # Local optimization with Nelder-Mead
    options = {
        'maxiter': 2000,
        'maxfev': 5000,
        'xatol': 1e-8,
        'fatol': 1e-10,
        'adaptive': True
    }
    
    for h0 in top_starts:
        if time.time() > deadline - 10:
            break
        
        try:
            res = minimize(_neg_value, h0, method='Nelder-Mead', options=options)
            if res.success or res.fun < -best_val:
                h_opt = np.maximum(res.x, 0.0)
                v_opt = _evaluate(h_opt)
                if v_opt > best_val:
                    best_val = v_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 5:
        try:
            options_tight = {
                'maxiter': 5000,
                'maxfev': 10000,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
            res = minimize(_neg_value, best_h, method='Nelder-Mead', options=options_tight)
            h_opt = np.maximum(res.x, 0.0)
            v_opt = _evaluate(h_opt)
            if v_opt > best_val:
                best_val = v_opt
                best_h = h_opt.copy()
        except Exception:
            pass
    
    return list(best_h)
# EVOLVE-BLOCK-END
