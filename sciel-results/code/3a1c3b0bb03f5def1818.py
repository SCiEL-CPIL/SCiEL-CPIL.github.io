# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _area_ref(x, n=64, M=20000):
    """Correct area via full sort at each height. O(M n log n)."""
    m = np.arange(M)
    y = (m + 0.5) / M
    i_idx = np.arange(n)
    a = (i_idx + 1) / n
    w = (1.0 - y) / n
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    Ls = np.sort(L, axis=0)
    d = Ls[1:, :] - Ls[:-1, :]
    union = w[np.newaxis, :] + np.sum(np.minimum(d, w[np.newaxis, :]), axis=0)
    return float(np.mean(union))

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    M = 20000
    start = time.time()
    deadline = start + 600.0

    i_idx = np.arange(n)
    a = (i_idx + 1) / n

    def objective(x):
        return _area_ref(x, n, M)

    # Seed candidates
    seeds = [
        -a * 0.5,
        np.zeros(n, dtype=np.float64),
        -a * 0.4,
        -a * 0.6,
        -a * 0.5 + np.random.default_rng(1).normal(0, 0.1, n),
    ]
    
    best_x = None
    best_val = np.inf

    for s in seeds:
        val = objective(s)
        if val < best_val:
            best_val = val
            best_x = s.copy()
            
        if time.time() > deadline - 30:
            break

    # Optimize from the best seed
    if best_x is not None and time.time() < deadline - 30:
        # Use a smaller M for optimization to speed up?
        # _area_ref uses M=20000.
        # Let's create a wrapper that uses M=2000 for speed during optimization
        M_opt = 2000
        def objective_opt(x):
            return _area_ref(x, n, M_opt)
            
        res = minimize(objective_opt, best_x, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
        
        if res.success and res.fun < best_val:
            # Refine with full M
            x_refined = res.x
            # Do a few more steps with full M?
            # Or just evaluate.
            val_full = objective(x_refined)
            if val_full < best_val:
                best_val = val_full
                best_x = x_refined.copy()

    # Final validation
    if best_x is None or not np.all(np.isfinite(best_x)):
        return np.zeros(n, dtype=np.float64)
    
    # One last check with full M
    final_val = _area_ref(best_x, n, M)
    if not np.isfinite(final_val) or final_val <= 0:
        return np.zeros(n, dtype=np.float64)
        
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
