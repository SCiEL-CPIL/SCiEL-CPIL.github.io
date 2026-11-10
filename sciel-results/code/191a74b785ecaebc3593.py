# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar


def _union_area_and_grad(x, M=4000):
    """
    Compute the union area of 64 triangles and its gradient w.r.t. x.
    
    x: shape (64,) - base positions
    M: number of quadrature points
    Returns: (area, grad) where grad has shape (64,)
    """
    n = 64
    a = (np.arange(n) + 1) / n  # a_i = (i+1)/64
    y = (np.arange(M) + 0.5) / M  # shape (M,)
    
    # L_i(y) = x_i + a_i * y, shape (M, n)
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n  # shape (M,)
    
    # Sort L along axis=1 and track permutation
    order = np.argsort(L, axis=1)  # shape (M, n)
    L_sorted = np.take_along_axis(L, order, axis=1)  # shape (M, n)
    
    # Consecutive differences
    diffs = np.diff(L_sorted, axis=1)  # shape (M, n-1)
    
    # mask: (M, n-1) boolean, True where diffs < w
    mask = diffs < w[:, np.newaxis]  # shape (M, n-1)
    
    # union_len at each y
    union_len = w + np.sum(np.minimum(diffs, w[:, np.newaxis]), axis=1)
    area = np.mean(union_len)
    
    # Gradient computation using bincount for speed
    grad = np.zeros(n, dtype=np.float64)
    
    # For each j in 0..n-2, where mask[m, j] is True:
    #   +1 to grad[order[m, j+1]]
    #   -1 to grad[order[m, j]]
    
    # Vectorized approach using bincount
    # Flatten the mask and order arrays
    mask_flat = mask.ravel()  # shape (M*(n-1),)
    order_flat = order.ravel()  # shape (M*n,)
    
    # For each position in the flattened mask, we need to map to the corresponding
    # left and right indices in order
    # mask has shape (M, n-1), order has shape (M, n)
    # For mask[m, j], left = order[m, j], right = order[m, j+1]
    
    # Create indices
    m_idx, j_idx = np.where(mask)  # m_idx: rows, j_idx: cols
    if len(m_idx) > 0:
        left_indices = order[m_idx, j_idx]
        right_indices = order[m_idx, j_idx + 1]
        
        # Use bincount for efficient accumulation
        grad += np.bincount(right_indices, minlength=n).astype(np.float64)
        grad -= np.bincount(left_indices, minlength=n).astype(np.float64)
    
    grad /= M  # mean over quadrature points
    
    return area, grad


def _union_area_only(x, M=20000):
    """
    Compute the union area of 64 triangles (no gradient).
    Uses full quadrature for final evaluation.
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    
    L_sorted = np.sort(L, axis=1)
    diffs = np.diff(L_sorted, axis=1)
    
    union_len = w + np.sum(np.minimum(diffs, w[:, np.newaxis]), axis=1)
    area = np.mean(union_len)
    
    return area


def _keich_init(n=64):
    """Generate Keich construction initial guess."""
    k = int(np.log2(n))
    j_arr = np.arange(1, k + 1)
    positions = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)
    return positions


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Uses the Keich construction as initialization, then refines with L-BFGS-B
    using the analytical gradient of the union area, followed by Nelder-Mead
    polish to handle the non-smooth objective.
    
    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    start_time = time.time()
    deadline = start_time + 630.0  # 630 second limit (evaluator kills at 630s)
    safety_margin = 15.0  # seconds before deadline
    
    def time_left():
        return deadline - time.time() - safety_margin
    
    # Keich construction as initial guess
    best_x = _keich_init(n).copy()
    best_area = _union_area_and_grad(best_x)[0]
    
    # Round 1: L-BFGS-B with moderate quadrature
    if time_left() > 60:
        try:
            result = minimize(
                _union_area_and_grad,
                best_x,
                method='L-BFGS-B',
                jac=True,
                args=(4000,),
                options={
                    'maxiter': 300,
                    'maxfun': 800,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 20,
                }
            )
            
            if result.success or result.fun < best_area:
                candidate_x = result.x
                if np.all(np.isfinite(candidate_x)):
                    area_candidate, _ = _union_area_and_grad(candidate_x)
                    if np.isfinite(area_candidate) and area_candidate > 0 and area_candidate < best_area:
                        best_x = candidate_x
                        best_area = area_candidate
        except Exception:
            pass
    
    # Round 2: Nelder-Mead polish on the non-smooth objective
    if time_left() > 30:
        try:
            # Use a slightly coarser quadrature for NM to keep it fast
            nm_func = lambda x: _union_area_only(x, M=8000)
            result_nm = minimize(
                nm_func,
                best_x,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True,
                }
            )
            
            if result_nm.success or result_nm.fun < best_area:
                candidate_x = result_nm.x
                if np.all(np.isfinite(candidate_x)):
                    area_candidate = _union_area_only(candidate_x, M=20000)
                    if np.isfinite(area_candidate) and area_candidate > 0 and area_candidate < best_area:
                        best_x = candidate_x
                        best_area = area_candidate
        except Exception:
            pass
    
    # Round 3: Another L-BFGS-B pass with finer quadrature
    if time_left() > 60:
        try:
            result2 = minimize(
                _union_area_and_grad,
                best_x,
                method='L-BFGS-B',
                jac=True,
                args=(8000,),
                options={
                    'maxiter': 200,
                    'maxfun': 500,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'maxls': 20,
                }
            )
            
            if result2.success or result2.fun < best_area:
                candidate_x = result2.x
                if np.all(np.isfinite(candidate_x)):
                    area_candidate, _ = _union_area_and_grad(candidate_x, M=8000)
                    if np.isfinite(area_candidate) and area_candidate > 0 and area_candidate < best_area:
                        best_x = candidate_x
                        best_area = area_candidate
        except Exception:
            pass
    
    # Round 4: Final Nelder-Mead polish with full quadrature
    if time_left() > 30:
        try:
            nm_func_final = lambda x: _union_area_only(x, M=20000)
            result_nm2 = minimize(
                nm_func_final,
                best_x,
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-9,
                    'fatol': 1e-12,
                    'adaptive': True,
                }
            )
            
            if result_nm2.success or result_nm2.fun < best_area:
                candidate_x = result_nm2.x
                if np.all(np.isfinite(candidate_x)):
                    area_candidate = _union_area_only(candidate_x, M=20000)
                    if np.isfinite(area_candidate) and area_candidate > 0 and area_candidate < best_area:
                        best_x = candidate_x
                        best_area = area_candidate
        except Exception:
            pass
    
    # Final validation
    if not np.all(np.isfinite(best_x)):
        best_x = _keich_init(n)
    
    return best_x


# EVOLVE-BLOCK-END
