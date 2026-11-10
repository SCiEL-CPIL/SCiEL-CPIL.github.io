import numpy as np
from scipy.optimize import minimize
import time

# Precompute constants
N = 64
M = 20000
Y = (np.arange(M) + 0.5) / M
A = (np.arange(N) + 1) / N
W = (1.0 - Y) / N

def _area_exact(x):
    """
    Computes the exact area of the union of triangles.
    x: shape (N,)
    """
    # L[i, m] = x[i] + A[i] * Y[m]
    L = x[:, np.newaxis] + A[:, np.newaxis] * Y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    diffs = np.diff(L_sorted, axis=0)
    # min(diffs, W)
    mins = np.minimum(diffs, W[np.newaxis, :])
    union_len = W[np.newaxis, :] + np.sum(mins, axis=0)
    return np.mean(union_len)

def _area_smooth(x, tau=50.0):
    """
    Computes a smooth approximation of the area using log-sum-exp for min.
    min(a, b) = -max(-a, -b) = -1/tau * log(exp(-tau*a) + exp(-tau*b))
    """
    L = x[:, np.newaxis] + A[:, np.newaxis] * Y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    diffs = np.diff(L_sorted, axis=0)
    
    # We need sum over j of min(diffs[j], W)
    # Let d = diffs, w = W
    # min(d, w) = -1/tau * log(exp(-tau*d) + exp(-tau*w))
    
    # To avoid overflow/underflow, use logsumexp trick
    # log(exp(-tau*d) + exp(-tau*w)) = -tau*max(d,w) + log(exp(-tau*(d-max)) + exp(-tau*(w-max)))
    # This is getting complicated to vectorize correctly for the sum.
    # Simpler:
    # term = np.minimum(diffs, W[np.newaxis, :]) is the exact one.
    # Smooth:
    # val = -1.0/tau * np.logaddexp(-tau * diffs, -tau * W[np.newaxis, :])
    
    val = -1.0 / tau * np.logaddexp(-tau * diffs, -tau * W[np.newaxis, :])
    union_len = W[np.newaxis, :] + np.sum(val, axis=0)
    return np.mean(union_len)

def _keich_positions():
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)

def kakeya_needle_2d_positions64() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin
    
    best_x = _keich_positions()
    best_score = _area_exact(best_x)
    
    # Strategy:
    # 1. Polish the Keich solution with Nelder-Mead on the exact function.
    # 2. Try optimizing the smooth surrogate from Keich and random starts, then polish.
    
    def _try_optimize(x0, method, options):
        nonlocal best_x, best_score
        try:
            res = minimize(_area_exact, x0, method=method, options=options)
            if res.success or res.fun < best_score:
                if np.all(np.isfinite(res.x)):
                    score = _area_exact(res.x)
                    if score < best_score:
                        best_score = score
                        best_x = res.x.copy()
        except Exception:
            pass

    # 1. Nelder-Mead on Exact from Keich
    if time.time() < deadline:
        _try_optimize(best_x, 'Nelder-Mead', {'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12})

    # 2. Smooth Surrogate Optimization
    # Use a few different taus
    taus = [10.0, 50.0, 100.0]
    for tau in taus:
        if time.time() > deadline: break
        
        # Optimize smooth from Keich
        x0 = best_x.copy()
        try:
            res_smooth = minimize(lambda x: _area_smooth(x, tau), x0, method='Nelder-Mead', 
                                  options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            x_candidate = res_smooth.x
            # Polish with exact
            _try_optimize(x_candidate, 'Nelder-Mead', {'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
        except Exception:
            pass

        # Optimize smooth from random
        for _ in range(5):
            if time.time() > deadline: break
            x_rand = np.random.uniform(-1, 1, N)
            try:
                res_smooth = minimize(lambda x: _area_smooth(x, tau), x_rand, method='Nelder-Mead', 
                                      options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
                x_candidate = res_smooth.x
                _try_optimize(x_candidate, 'Nelder-Mead', {'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            except Exception:
                pass

    # 3. Random Restarts on Exact
    while time.time() < deadline:
        x_rand = np.random.uniform(-1, 1, N)
        _try_optimize(x_rand, 'Nelder-Mead', {'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})

    return best_x
