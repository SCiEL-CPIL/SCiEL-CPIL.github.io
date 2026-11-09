# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _union_area(x, M=20000):
    """
    Compute the union area of 64 triangles.
    
    x: shape (64,) - base positions
    M: number of quadrature points
    Returns: area (scalar)
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    
    # L_i(y) = x_i + a_i * y
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    
    # Sort L along axis=1
    order = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, order, axis=1)
    
    # Consecutive differences
    diffs = np.diff(L_sorted, axis=1)
    
    # union_len at each y
    union_len = w + np.sum(np.minimum(diffs, w[:, np.newaxis]), axis=1)
    area = np.mean(union_len)
    
    return area


def _keich_init(n=64):
    """
    Keich construction initial guess.
    """
    k = int(np.log2(n))
    j_arr = np.arange(1, k + 1)
    positions = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)
    return positions


def _random_init(n=64, seed=0):
    """
    Random initial guess.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, size=n)


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Uses a multi-restart strategy with Nelder-Mead local optimization.
    Nelder-Mead is robust to non-smoothness in the objective function.
    
    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit
    safety_margin = 10.0  # seconds before deadline
    
    best_x = _keich_init(n)
    best_area = _union_area(best_x, M=2000)
    
    # Define the objective function for the optimizer
    # We use M=2000 for speed during optimization, then verify with M=20000
    def objective(x):
        return _union_area(x, M=2000)
    
    # Multiple restarts
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline - safety_margin:
            break
        
        if restart == 0:
            x0 = _keich_init(n)
        elif restart % 3 == 1:
            # Perturbed Keich
            rng = np.random.default_rng(restart)
            x0 = _keich_init(n) + rng.normal(0, 0.05, size=n)
        else:
            # Random
            x0 = _random_init(n, seed=restart)
        
        # Run Nelder-Mead
        try:
            res = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            x_opt = res.x
            area_opt = _union_area(x_opt, M=2000)
            
            # Verify with higher resolution if it looks promising
            if area_opt < best_area + 1e-4:
                area_high_res = _union_area(x_opt, M=20000)
                if area_high_res < best_area:
                    best_x = x_opt.copy()
                    best_area = area_high_res
        except Exception:
            continue
    
    return best_x


# EVOLVE-BLOCK-END
