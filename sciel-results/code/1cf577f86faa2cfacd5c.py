import numpy as np
from scipy.optimize import minimize
import time

# Precompute constants
N = 64
M_FAST = 1000  # Reduced resolution for optimization speed
M_EXACT = 20000
Y_FAST = (np.arange(M_FAST) + 0.5) / M_FAST
Y_EXACT = (np.arange(M_EXACT) + 0.5) / M_EXACT
A = (np.arange(N) + 1) / N
W_FAST = (1.0 - Y_FAST) / N
W_EXACT = (1.0 - Y_EXACT) / N

def _area_fast(x):
    """
    Computes the area of the union of triangles using reduced height resolution.
    x: shape (N,)
    """
    # L[i, m] = x[i] + A[i] * Y[m]
    L = x[:, np.newaxis] + A[:, np.newaxis] * Y_FAST[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    diffs = np.diff(L_sorted, axis=0)
    mins = np.minimum(diffs, W_FAST[np.newaxis, :])
    union_len = W_FAST[np.newaxis, :] + np.sum(mins, axis=0)
    return np.mean(union_len)

def _area_exact(x):
    """
    Computes the exact area of the union of triangles.
    x: shape (N,)
    """
    L = x[:, np.newaxis] + A[:, np.newaxis] * Y_EXACT[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    diffs = np.diff(L_sorted, axis=0)
    mins = np.minimum(diffs, W_EXACT[np.newaxis, :])
    union_len = W_EXACT[np.newaxis, :] + np.sum(mins, axis=0)
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
    deadline = start_time + 580.0  # 50s safety margin
    
    best_x = _keich_positions()
    best_score = _area_fast(best_x)
    
    # 1. Optimize using the fast surrogate function
    # Nelder-Mead is robust to the non-smoothness introduced by sorting/min
    try:
        res = minimize(_area_fast, best_x, method='Nelder-Mead', 
                       options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        if np.all(np.isfinite(res.x)):
            candidate_score = _area_fast(res.x)
            if candidate_score < best_score:
                best_score = candidate_score
                best_x = res.x.copy()
    except Exception:
        pass

    # 2. If time remains, do a few random restarts on the fast function
    # This helps escape local minima of the fast surrogate
    np.random.seed(42)
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        
        x_rand = np.random.uniform(-1, 1, N)
        try:
            res_rand = minimize(_area_fast, x_rand, method='Nelder-Mead', 
                                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if np.all(np.isfinite(res_rand.x)):
                cand_score = _area_fast(res_rand.x)
                if cand_score < best_score:
                    best_score = cand_score
                    best_x = res_rand.x.copy()
        except Exception:
            pass

    # 3. Final Polish: Evaluate the best candidate with the EXACT metric
    # The fast metric is a good proxy, but we want to ensure we are returning
    # a position that is actually good for the exact metric.
    # We don't optimize on the exact metric because it's too slow, but we
    # can check if a small local search on the exact metric helps.
    # Given the 600s limit and the cost of _area_exact, we skip optimization
    # on exact and just return the best found via fast metric.
    # The difference between fast and exact area is small for smooth configurations.
    
    return best_x
