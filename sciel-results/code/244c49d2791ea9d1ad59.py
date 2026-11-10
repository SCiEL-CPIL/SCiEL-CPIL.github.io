# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _calc_area(x):
    """
    Compute the area of the union of 64 triangles defined by base positions x.
    x: shape (64,)
    Returns: scalar area
    """
    n = 64
    y = np.linspace(0.5 / 20000, 1 - 0.5 / 20000, 20000)
    a = (np.arange(n) + 1) / n
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1 - y) / n
    L_sorted = np.sort(L, axis=0)
    gaps = np.diff(L_sorted, axis=0)
    contrib = np.minimum(gaps, w[None, :])
    union_len = w + contrib.sum(axis=0)
    return union_len.mean()

def kakeya_needle_2d_positions64() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 630  # Safety margin
    
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    
    # Initialize with Keich construction
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0 = np.array(positions, dtype=np.float64)
    
    best_x = x0.copy()
    best_area = _calc_area(best_x)
    
    # Run Nelder-Mead optimization
    # We use a custom options setup to allow many iterations
    # The budget is 630s, so we set maxiter high and rely on the time guard
    # However, scipy.optimize.minimize doesn't have a built-in time limit.
    # We will run it in a loop or just set a very high maxiter and check time periodically?
    # Actually, scipy minimize doesn't support time limits directly.
    # A common trick is to use a callback that raises an exception or checks time.
    # But for robustness, we can just run a single minimize call with a high maxiter
    # and hope it finishes, or use a simpler gradient-free method we control.
    
    # Let's use a manual Nelder-Mead implementation or just call scipy with a callback
    # that checks time.
    
    def time_check_callback(xk):
        nonlocal best_x, best_area
        current_area = _calc_area(xk)
        if current_area < best_area:
            best_area = current_area
            best_x = xk.copy()
        if time.time() > deadline - 10:  # 10s safety margin
            raise TimeoutError("Time limit exceeded")
        
    try:
        # Nelder-Mead parameters
        # initial_simplex: we can let scipy generate it, or provide one.
        # Let's let scipy generate it based on x0.
        res = minimize(
            _calc_area, 
            x0, 
            method='Nelder-Mead', 
            options={
                'maxiter': 100000, 
                'xatol': 1e-10, 
                'fatol': 1e-10,
                'adaptive': True
            },
            callback=time_check_callback
        )
        if res.fun < best_area:
            best_area = res.fun
            best_x = res.x
    except TimeoutError:
        pass
    except Exception:
        # If anything goes wrong, return the best found so far
        pass
    
    # Ensure the result is valid
    if not np.all(np.isfinite(best_x)):
        return x0
        
    return best_x

# EVOLVE-BLOCK-END
