# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_area_and_grad(x):
    n = 64
    M = 20000
    a = np.arange(1, n + 1) / n
    y = (np.arange(M) + 0.5) / M
    w = (1 - y) / n
    
    # L shape (M, n)
    # L[m, i] = x[i] + a[i] * y[m]
    L = x[None, :] + a[None, :] * y[:, None]
    
    # Sort L along axis 1
    # indices shape (M, n)
    sort_idx = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, sort_idx, axis=1)
    
    # d shape (M, n-1)
    d = L_sorted[:, 1:] - L_sorted[:, :-1]
    
    # Union length U shape (M,)
    # U = w + sum(min(d, w))
    # min(d, w) where w is broadcast
    w_col = w[:, None]
    d_clipped = np.minimum(d, w_col)
    U = w + np.sum(d_clipped, axis=1)
    
    area = np.mean(U)
    
    # Gradient
    # g shape (M, n)
    # Condition d < w
    mask = d < w_col
    
    # Flatten indices
    m_indices = np.where(mask)[0]
    j_indices = np.where(mask)[1]
    
    # grad_L = np.zeros_like(L)
    # np.add.at(grad_L, (m_indices, sort_idx[m_indices, j_indices]), -1.0)
    # np.add.at(grad_L, (m_indices, sort_idx[m_indices, j_indices + 1]), 1.0)
    
    # More efficient gradient computation using bincount or similar?
    # Let's stick to add.at for correctness, it's fast enough for n=64, M=20000
    # Actually, let's try to optimize the gradient calculation.
    
    # grad_L[m, i] = -1 if i is sort_idx[m, j] for some j where mask[m,j] is true
    #               +1 if i is sort_idx[m, j+1] for some j where mask[m,j] is true
    
    # We can create an array of indices to decrement and increment
    # Decrement indices: (m_indices, sort_idx[m_indices, j_indices])
    # Increment indices: (m_indices, sort_idx[m_indices, j_indices + 1])
    
    # Using np.add.at is safe but can be slow.
    # Alternative:
    # grad_L = np.zeros_like(L)
    # np.add.at(grad_L, (m_indices, sort_idx[m_indices, j_indices]), -1.0)
    # np.add.at(grad_L, (m_indices, sort_idx[m_indices, j_indices + 1]), 1.0)
    
    # Let's use this standard approach.
    grad_L = np.zeros_like(L)
    np.add.at(grad_L, (m_indices, sort_idx[m_indices, j_indices]), -1.0)
    np.add.at(grad_L, (m_indices, sort_idx[m_indices, j_indices + 1]), 1.0)
    
    grad_x = np.sum(grad_L, axis=0) / M
    
    return area, grad_x

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0 = np.array(positions, dtype=np.float64)
    
    # Optimize
    start_time = time.time()
    deadline = start_time + 590 # 10s margin
    
    # First pass: L-BFGS-B with analytic gradient
    result = minimize(
        lambda x: _compute_area_and_grad(x)[0],
        x0,
        jac=lambda x: _compute_area_and_grad(x)[1],
        method='L-BFGS-B',
        options={'maxiter': 2000, 'maxfun': 20000, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    best_x = result.x
    best_area = _compute_area_and_grad(best_x)[0]
    
    # Second pass: Coordinate descent polishing
    # The function is convex, so coordinate descent can find the exact minimum
    # if we are careful. However, L-BFGS-B should be close.
    # Let's try a few rounds of coordinate descent with minimize_scalar
    # or just simple line search.
    
    # Given the time limit, let's see if we have time for more refinement.
    if time.time() < deadline - 60:
        # Try a second round of L-BFGS-B starting from the previous result
        result2 = minimize(
            lambda x: _compute_area_and_grad(x)[0],
            best_x,
            jac=lambda x: _compute_area_and_grad(x)[1],
            method='L-BFGS-B',
            options={'maxiter': 2000, 'maxfun': 20000, 'disp': False, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result2.fun < best_area:
            best_x = result2.x
            best_area = result2.fun
    
    # Ensure validity
    if not np.all(np.isfinite(best_x)):
        best_x = x0.copy()
        
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
