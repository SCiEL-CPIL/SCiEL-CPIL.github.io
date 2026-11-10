import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def compute_M(f, N=8):
    f = np.asarray(f, dtype=np.float64)
    breaks = [0.0, 1.0]
    invN = 1.0 / N
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            for offset in [0.0, invN, -invN]:
                c_val = (f[j] - f[i] + offset) * N / (i - j)
                if 0.0 < c_val < 1.0:
                    breaks.append(c_val)
    breaks = sorted(set(breaks))
    total = 0.0
    for k in range(len(breaks) - 1):
        c_mid = (breaks[k] + breaks[k+1]) / 2.0
        intervals = []
        for i in range(N):
            lo = f[i] + c_mid * i * invN
            hi = lo + invN
            intervals.append((lo, hi))
        intervals.sort()
        merged_len = 0.0
        cur_lo, cur_hi = intervals[0]
        for lo, hi in intervals[1:]:
            if lo <= cur_hi:
                if hi > cur_hi:
                    cur_hi = hi
            else:
                merged_len += cur_hi - cur_lo
                cur_lo, cur_hi = lo, hi
        merged_len += cur_hi - cur_lo
        total += merged_len * (breaks[k+1] - breaks[k])
    return total

def objective(f):
    return compute_M(f)

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 600.0
    
    # Keich baseline
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    best_f = keich.copy()
    best_M = compute_M(best_f)
    
    # Try to improve with optimization
    # Use differential evolution first (global), then local refinement
    rng = np.random.default_rng(42)
    
    # Bounds: f values should be in a reasonable range
    # The intervals at c=0 are [f(i), f(i)+1/N], at c=1 are [f(i)+i/N, f(i)+i/N+1/N]
    # For the union to be small, f should be chosen so intervals overlap a lot
    # Reasonable range: [-1, 2] or so
    bounds = [(-2.0, 2.0)] * N
    
    # Try a few optimization methods
    methods = [
        ('differential_evolution', {}),
        ('nelder-mead', {'x0': best_f.copy(), 'maxiter': 5000, 'adaptive': True}),
    ]
    
    for method_name, kwargs in methods:
        if time.time() > deadline - 30:
            break
        if method_name == 'differential_evolution':
            try:
                result = differential_evolution(
                    objective, bounds, seed=42, maxiter=100, tol=1e-10,
                    polish=True, workers=-1
                )
                if result.fun < best_M:
                    best_M = result.fun
                    best_f = result.x.copy()
            except Exception:
                pass
        elif method_name == 'nelder-mead':
            try:
                result = minimize(
                    objective, kwargs['x0'], method='Nelder-Mead',
                    options={'maxiter': 5000, 'adaptive': True, 'xatol': 1e-12, 'fatol': 1e-12}
                )
                if result.fun < best_M:
                    best_M = result.fun
                    best_f = result.x.copy()
            except Exception:
                pass
    
    # Local refinement from multiple starting points
    if time.time() < deadline - 30:
        for start in range(20):
            if time.time() > deadline - 30:
                break
            if start == 0:
                x0 = best_f.copy()
            else:
                x0 = best_f + rng.normal(0, 0.1, N)
            try:
                result = minimize(
                    objective, x0, method='Nelder-Mead',
                    options={'maxiter': 3000, 'adaptive': True, 'xatol': 1e-14, 'fatol': 1e-14}
                )
                if result.fun < best_M:
                    best_M = result.fun
                    best_f = result.x.copy()
            except Exception:
                pass
    
    # Build positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    base_x = best_f[i_flat]
    base_y = best_f[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions
