"""Optimized program for second autocorrelation inequality."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _evaluate(h):
    """Compute the value for a given height vector."""
    h = np.asarray(h, dtype=np.float64)
    g = np.convolve(h, h, mode='full')
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    y_sq = y**2
    cross = y * np.roll(y, -1)
    l2 = (w / 3.0) * np.sum(y_sq[:-1] + cross[:-1] + y_sq[1:])
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf < 1e-30:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _negative_value(h):
    return -_evaluate(h)

def _make_symmetric(h_asym):
    """Create a symmetric height vector from the first half."""
    n = len(h_asym)
    if n % 2 == 0:
        # Even: h = [a_0, ..., a_{n-1}, a_{n-1}, ..., a_0]
        h = np.concatenate([h_asym, h_asym[::-1]])
    else:
        # Odd: h = [a_0, ..., a_{n-1}, ..., a_0]
        h = np.concatenate([h_asym, h_asym[-2::-1]])
    return h

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    # Try various initial configurations
    inits = []
    
    # 1. Uniform (seed)
    inits.append(np.ones(N))
    
    # 2. Single peak at center
    for peak_width in [1, 2, 3, 4, 5]:
        h = np.zeros(N)
        center = N // 2
        for i in range(-peak_width//2, peak_width//2 + 1):
            idx = center + i
            if 0 <= idx < N:
                h[idx] = 1.0
        inits.append(h)
    
    # 3. Gaussian-like profiles
    for sigma in [2, 3, 4, 5, 6, 8, 10]:
        x = np.arange(N) - N/2.0
        h = np.exp(-0.5 * (x/sigma)**2)
        h = h / np.max(h) if np.max(h) > 0 else h
        inits.append(h)
    
    # 4. Bump functions (compact support)
    for width in [4, 6, 8, 10, 12, 15]:
        h = np.zeros(N)
        center = N // 2
        half_w = width // 2
        for i in range(center - half_w, center + half_w + 1):
            if 0 <= i < N:
                t = (i - center) / half_w
                h[i] = np.cos(np.pi * t / 2)**2
        if np.max(h) > 0:
            h = h / np.max(h)
        inits.append(h)
    
    # 5. Triangle / tent functions
    for width in [4, 6, 8, 10, 12, 15, 20]:
        h = np.zeros(N)
        center = N // 2
        half_w = width // 2
        for i in range(center - half_w, center + half_w + 1):
            if 0 <= i < N:
                t = abs(i - center) / half_w
                h[i] = max(0, 1 - t)
        inits.append(h)
    
    # 6. Sine bump
    for width in [8, 12, 16, 20, 25]:
        h = np.zeros(N)
        center = N // 2
        half_w = width // 2
        for i in range(center - half_w, center + half_w + 1):
            if 0 <= i < N:
                t = (i - center) / half_w
                h[i] = np.sin(np.pi * (t + 1) / 2)**2
        inits.append(h)
    
    # 7. Two separated peaks
    for sep in [4, 6, 8, 10]:
        for peak_w in [2, 3, 4]:
            h = np.zeros(N)
            center = N // 2
            for peak_center in [center - sep, center + sep]:
                for i in range(-peak_w//2, peak_w//2 + 1):
                    idx = peak_center + i
                    if 0 <= idx < N:
                        h[idx] = 1.0
            inits.append(h)
    
    # 8. Random perturbations of uniform
    rng = np.random.RandomState(42)
    for _ in range(5):
        h = np.ones(N)
        h += rng.normal(0, 0.3, N)
        h = np.maximum(h, 0)
        inits.append(h)
    
    # Evaluate all initial points
    for h_init in inits:
        if time.time() > deadline - 5.0:
            break
        val = _evaluate(h_init)
        if val > best_val:
            best_val = val
            best_h = h_init.copy()
    
    # L-BFGS-B optimization from best initial point and a few others
    # Use full h (non-symmetric) for more flexibility, with bounds [0, inf)
    bounds = [(0.0, None)] * N
    
    # Sort inits by value descending
    init_vals = [(h, _evaluate(h)) for h in inits]
    init_vals.sort(key=lambda x: -x[1])
    
    # Optimize from top initial points
    num_opt = min(10, len(init_vals))
    for i in range(num_opt):
        if time.time() > deadline - 30.0:
            break
        h0, _ = init_vals[i]
        try:
            res = minimize(
                _negative_value,
                h0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000}
            )
            if res.success or res.fun < -best_val:
                h_opt = np.maximum(res.x, 0)
                val_opt = _evaluate(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Polish: try a few more local optimizations with tighter tolerances from best
    if time.time() < deadline - 20.0:
        try:
            res = minimize(
                _negative_value,
                best_h,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 50000}
            )
            if res.fun < -best_val:
                h_opt = np.maximum(res.x, 0)
                val_opt = _evaluate(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final check
    final_val = _evaluate(best_h)
    if final_val > best_val:
        best_val = final_val
    
    return list(best_h)

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
