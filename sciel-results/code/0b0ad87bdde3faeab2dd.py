# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _union_area(x, n=64, M=20000):
    """
    Compute the area of the union of n triangles using the specified quadrature.
    
    Parameters:
    -----------
    x : array of shape (n,)
        Base positions x_i
    n : int
        Number of triangles (default 64)
    M : int
        Number of quadrature heights (default 20000)
    
    Returns:
    --------
    area : float
        The area of the union
    """
    # Heights y_m = (m + 0.5) / M
    m = np.arange(M)
    y = (m + 0.5) / M  # shape (M,)
    
    # a_i = (i + 1) / n
    i_idx = np.arange(n)
    a = (i_idx + 1) / n  # shape (n,)
    
    # w = (1 - y) / n
    w = (1.0 - y) / n  # shape (M,)
    
    # L_i(y) = x_i + a_i * y
    # L has shape (n, M)
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # Sort L along axis 0 (over n triangles)
    L_sorted = np.sort(L, axis=0)  # shape (n, M)
    
    # Compute consecutive differences
    # d_j = L_sorted[j+1] - L_sorted[j] for j = 0..n-2
    d = L_sorted[1:, :] - L_sorted[:-1, :]  # shape (n-1, M)
    
    # Union length = w + sum over j of min(d_j, w)
    # min(d_j, w) where w is broadcast
    min_d_w = np.minimum(d, w[np.newaxis, :])  # shape (n-1, M)
    
    union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)  # shape (M,)
    
    area = np.mean(union_lengths)
    return area


def _union_area_gradient(x, n=64, M=20000):
    """
    Compute the area and its gradient with respect to x.
    
    Returns:
    --------
    area : float
    grad : array of shape (n,)
    """
    # Heights
    m = np.arange(M)
    y = (m + 0.5) / M  # shape (M,)
    
    i_idx = np.arange(n)
    a = (i_idx + 1) / n  # shape (n,)
    
    w = (1.0 - y) / n  # shape (M,)
    
    # L_i(y) = x_i + a_i * y
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]  # shape (n, M)
    
    # Sort L and track which triangle is at each position
    sort_idx = np.argsort(L, axis=0)  # shape (n, M)
    L_sorted = np.take_along_axis(L, sort_idx, axis=0)  # shape (n, M)
    
    # Consecutive differences
    d = L_sorted[1:, :] - L_sorted[:-1, :]  # shape (n-1, M)
    
    # min(d_j, w)
    min_d_w = np.minimum(d, w[np.newaxis, :])
    
    union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
    area = np.mean(union_lengths)
    
    # Gradient computation
    # d(area)/d(x_i) = (1/M) * sum_m d(union_length_m)/d(x_i)
    # 
    # union_length_m = w_m + sum_j min(d_j(m), w_m)
    # 
    # d_j(m) = L_sorted[j+1](m) - L_sorted[j](m)
    # 
    # The derivative of min(d, w) w.r.t. d is 1 if d < w, 0 if d > w
    # 
    # d(d_j)/d(x_i) = d(L_sorted[j+1])/d(x_i) - d(L_sorted[j])/d(x_i)
    # 
    # L_sorted[j](m) = L_{sort_idx[j,m]}(m)
    # d(L_k(m))/d(x_i) = delta_{k,i}
    # 
    # So d(d_j(m))/d(x_i) = delta_{sort_idx[j+1,m], i} - delta_{sort_idx[j,m], i}
    
    # Create gradient accumulator
    grad = np.zeros(n, dtype=np.float64)
    
    # For each m and j, if d_j(m) < w_m, then:
    # grad[sort_idx[j+1,m]] += 1/M
    # grad[sort_idx[j,m]] -= 1/M
    
    # mask for where d_j < w
    mask = d < w[np.newaxis, :]  # shape (n-1, M)
    
    # For j = 0..n-2, m = 0..M-1
    # grad[sort_idx[j+1,m]] += 1/M where mask[j,m]
    # grad[sort_idx[j,m]] -= 1/M where mask[j,m]
    
    # Use np.add.at for efficient accumulation
    # sort_idx has shape (n, M)
    # For each j in 0..n-2:
    #   indices_upper = sort_idx[j+1, mask[j, :]]
    #   indices_lower = sort_idx[j, mask[j, :]]
    
    for j in range(n - 1):
        mask_j = mask[j, :]  # shape (M,)
        if np.any(mask_j):
            upper_idx = sort_idx[j + 1, mask_j]  # shape (num_true,)
            lower_idx = sort_idx[j, mask_j]  # shape (num_true,)
            np.add.at(grad, upper_idx, 1.0 / M)
            np.add.at(grad, lower_idx, -1.0 / M)
    
    return area, grad


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem.
    
    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    M = 20000
    
    deadline = time.time() + 600.0  # 600 second limit, use 580 for safety
    
    # Initial solution: Keich construction
    k = int(np.log2(n))
    j_arr = np.arange(1, k + 1)
    x0 = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)
    
    best_x = x0.copy()
    best_area = _union_area(x0, n, M)
    
    # Try optimization with time guard
    def objective(x):
        return _union_area(x, n, M)
    
    def objective_grad(x):
        area, grad = _union_area_gradient(x, n, M)
        return area, grad
    
    # Use L-BFGS-B with gradient
    try:
        result = minimize(
            objective_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'gtol': 1e-6, 'maxfun': 500}
        )
        
        if time.time() < deadline - 30:
            if result.fun < best_area:
                best_x = result.x.copy()
                best_area = result.fun
    except Exception:
        pass
    
    # If time remains, try more iterations or different starting points
    if time.time() < deadline - 60:
        # Try a few random perturbations
        np.random.seed(42)
        for restart in range(5):
            if time.time() > deadline - 30:
                break
            
            x_start = best_x + 0.1 * np.random.randn(n)
            try:
                result = minimize(
                    objective_grad,
                    x_start,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 100, 'gtol': 1e-6, 'maxfun': 200}
                )
                
                if result.fun < best_area:
                    best_x = result.x.copy()
                    best_area = result.fun
            except Exception:
                pass
    
    # Validate
    if not np.all(np.isfinite(best_x)):
        return x0
    if best_area <= 0:
        return x0
    
    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
