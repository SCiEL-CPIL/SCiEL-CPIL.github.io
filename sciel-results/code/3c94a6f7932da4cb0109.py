import numpy as np
from scipy.optimize import minimize
import time

# Precompute constants
N = 64
M_FAST = 1000
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

def _subgradient_fast(x):
    """
    Compute area and subgradient of the fast union area objective.
    Returns (area, grad) with grad shape (N,).
    """
    L = x[:, np.newaxis] + A[:, np.newaxis] * Y_FAST[np.newaxis, :]
    sort_idx = np.argsort(L, axis=0)
    
    i1 = sort_idx[:-1, :]  # (N-1, M_FAST)
    i2 = sort_idx[1:, :]   # (N-1, M_FAST)
    
    L_i1 = x[i1] + A[i1] * Y_FAST[np.newaxis, :]
    L_i2 = x[i2] + A[i2] * Y_FAST[np.newaxis, :]
    diffs = L_i2 - L_i1
    
    w_m = W_FAST[np.newaxis, :]
    active = diffs < w_m
    
    m_idx = np.repeat(np.arange(M_FAST), N - 1)
    grad_flat = np.zeros(N * M_FAST)
    
    neg_weight = active * (Y_FAST[np.newaxis, :] * A[i1])
    neg_flat_idx = i1.ravel() * M_FAST + m_idx
    np.add.at(grad_flat, neg_flat_idx, -neg_weight.ravel())
    
    pos_weight = active * (Y_FAST[np.newaxis, :] * A[i2])
    pos_flat_idx = i2.ravel() * M_FAST + m_idx
    np.add.at(grad_flat, pos_flat_idx, pos_weight.ravel())
    
    grad = np.mean(grad_flat.reshape(N, M_FAST), axis=1)
    
    mins = np.minimum(diffs, w_m)
    union_len = w_m + np.sum(mins, axis=0)
    area = np.mean(union_len)
    
    return area, grad

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
    deadline = start_time + 600.0
    
    best_x = _keich_positions()
    best_area = _area_fast(best_x)
    
    # Projected subgradient descent on the convex piecewise-linear objective
    x = best_x.copy()
    max_iter = 30000
    for t in range(1, max_iter + 1):
        if time.time() > deadline - 30:
            break
        
        area, grad = _subgradient_fast(x)
        
        if area < best_area:
            best_area = area
            best_x = x.copy()
        
        eta = 0.1 / np.sqrt(t)
        x = x - eta * grad
    
    # Final polish with Nelder-Mead on the fast surrogate
    if time.time() < deadline - 10:
        try:
            res = minimize(_area_fast, best_x, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            if np.all(np.isfinite(res.x)):
                cand_area = _area_fast(res.x)
                if cand_area < best_area:
                    best_area = cand_area
                    best_x = res.x.copy()
        except Exception:
            pass
    
    return best_x
