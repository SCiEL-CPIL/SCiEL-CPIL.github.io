# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _union_area_and_grad(x):
    """
    Compute the union area of 64 triangles and its gradient w.r.t. x.
    
    x: shape (64,) - base positions
    Returns: (area, grad) where grad has shape (64,)
    """
    n = 64
    a = (np.arange(n) + 1) / n  # a_i = (i+1)/64
    # Use fewer quadrature points for speed during optimization
    # The final evaluation uses 20000, but for optimization we can use fewer
    # and the result should still be good
    M = 4000  # quadrature points for optimization
    y = (np.arange(M) + 0.5) / M  # shape (M,)
    
    # L_i(y) = x_i + a_i * y, shape (M, n)
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n  # shape (M,)
    
    # Sort L along axis=1 and track permutation
    order = np.argsort(L, axis=1)  # shape (M, n)
    L_sorted = np.take_along_axis(L, order, axis=1)  # shape (M, n)
    
    # Consecutive differences
    diffs = np.diff(L_sorted, axis=1)  # shape (M, n-1)
    
    # For each consecutive pair, check if gap < w (i.e., they overlap/are close)
    # The union length = w + sum(min(diffs, w))
    # Gradient contribution: for pair (j, j+1) in sorted order,
    # if diffs[j] < w, then the term is diffs[j], and its gradient w.r.t. x
    # is (1 if x_{order[j+1]} is the "right" endpoint contribution)
    # Actually: d/dx_k [L_{sorted[j+1]} - L_{sorted[j]}] 
    #   = 1 if k == order[j+1], -1 if k == order[j], 0 otherwise
    # But we only include it if diffs[j] < w
    
    # For the base term w: d/dx_k [w] = 0 (w doesn't depend on x)
    # Wait, w = (1-y)/n, doesn't depend on x. So gradient of w is 0.
    # The gradient comes from sum(min(diffs, w))
    
    # mask: (M, n-1) boolean, True where diffs < w
    mask = diffs < w[:, np.newaxis]  # shape (M, n-1)
    
    # union_len at each y
    union_len = w + np.sum(np.minimum(diffs, w[:, np.newaxis]), axis=1)
    area = np.mean(union_len)
    
    # Gradient computation:
    # For each y_m and each consecutive pair j (0..n-2) in sorted order:
    #   if mask[m, j] is True, contribution to grad is:
    #     +1 to grad[order[m, j+1]]
    #     -1 to grad[order[m, j]]
    #   (because d(L_{sorted[j+1]} - L_{sorted[j]})/dx_k)
    # The total gradient is the mean over all m of these contributions.
    
    grad = np.zeros(n, dtype=np.float64)
    
    # For efficiency, we can use np.add.at or bincount
    # order has shape (M, n), we need order[:, j] and order[:, j+1] for each j
    # where mask[m, j] is True
    
    # Vectorized approach:
    # For each j in 0..n-2:
    #   idx_left = order[:, j] where mask[:, j]
    #   idx_right = order[:, j+1] where mask[:, j]
    #   grad[idx_right] += count
    #   grad[idx_left] -= count
    
    for j in range(n - 1):
        m_idx = np.where(mask[:, j])[0]
        if len(m_idx) > 0:
            left_indices = order[m_idx, j]
            right_indices = order[m_idx, j + 1]
            np.add.at(grad, right_indices, 1.0)
            np.add.at(grad, left_indices, -1.0)
    
    grad /= M  # mean over quadrature points
    
    return area, grad


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Uses the Keich construction as initialization, then refines with L-BFGS-B
    using the analytical gradient of the union area.
    
    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    k = int(np.log2(n))
    
    # Keich construction as initial guess
    j_arr = np.arange(1, k + 1)
    positions_keich = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions_keich[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)
    
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit
    safety_margin = 30.0  # seconds before deadline
    
    best_x = positions_keich.copy()
    best_area = _union_area_and_grad(best_x)[0]
    
    # Try L-BFGS-B optimization
    try:
        result = minimize(
            _union_area_and_grad,
            best_x,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 200,
                'maxfun': 500,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 20,
            }
        )
        
        if result.success or result.fun < best_area:
            candidate_x = result.x
            # Validate
            if np.all(np.isfinite(candidate_x)):
                area_candidate, _ = _union_area_and_grad(candidate_x)
                if np.isfinite(area_candidate) and area_candidate > 0 and area_candidate < best_area:
                    best_x = candidate_x
                    best_area = area_candidate
        
        # Check time
        if time.time() > deadline - safety_margin:
            return best_x
    except Exception:
        pass
    
    # If we have time, try a second round with more quadrature points
    if time.time() < deadline - safety_margin:
        try:
            # Use the refined result as new initial guess
            result2 = minimize(
                _union_area_and_grad,
                best_x,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 100,
                    'maxfun': 200,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'maxls': 20,
                }
            )
            
            if result2.success or result2.fun < best_area:
                candidate_x = result2.x
                if np.all(np.isfinite(candidate_x)):
                    area_candidate, _ = _union_area_and_grad(candidate_x)
                    if np.isfinite(area_candidate) and area_candidate > 0 and area_candidate < best_area:
                        best_x = candidate_x
                        best_area = area_candidate
        except Exception:
            pass
    
    return best_x


# EVOLVE-BLOCK-END
