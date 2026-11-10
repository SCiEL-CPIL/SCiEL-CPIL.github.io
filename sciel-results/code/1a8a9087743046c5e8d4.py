# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_area_and_grad(x, n_q=2000):
    """
    Compute area and gradient of the union area of 64 triangles.
    Uses n_q quadrature points for speed.
    """
    n = 64
    a = (np.arange(n) + 1) / n  # a_i = (i+1)/64
    
    m = np.arange(n_q)
    y = (m + 0.5) / n_q
    
    # L[m, i] = x[i] + a[i] * y[m]
    L = x[None, :] + a[None, :] * y[:, None]
    w = (1.0 - y) / 64.0  # shape (n_q,)
    
    # Sort L and get indices
    # idx[m, j] is the index of the j-th smallest L in row m
    idx = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, idx, axis=1)
    
    # Gaps d[m, j] = L_sorted[m, j+1] - L_sorted[m, j] for j=0..62
    d = np.diff(L_sorted, axis=1) # shape (n_q, 63)
    
    # Active mask: d < w
    # Note: at d == w, the gradient is ambiguous, but measure zero.
    active = d < w[:, None] # shape (n_q, 63)
    
    # Union length
    union_lengths = w + np.sum(np.where(active, d, w[:, None]), axis=1)
    area = np.mean(union_lengths)
    
    # Gradient calculation
    # grad[i] = sum over m, j where active[m,j] of (a[idx[m,j+1]] * delta(i, idx[m,j+1]) - a[idx[m,j]] * delta(i, idx[m,j]))
    # This can be vectorized.
    # For each active gap (m, j), we add a[idx[m,j+1]] to grad[idx[m,j+1]] and subtract a[idx[m,j]] from grad[idx[m,j]].
    
    # Flatten indices for advanced indexing
    # active shape (n_q, 63)
    # We need to scatter the contributions.
    
    # Let's use np.add.at for accumulation
    grad = np.zeros(n)
    
    # Get the indices of active gaps
    # active_rows, active_cols = np.where(active)
    # This might be sparse. Let's estimate sparsity.
    # w goes from ~1/64 to 0.
    # d is the gap between sorted L.
    # If triangles are well separated, d > w often.
    # If they overlap, d < w.
    
    # To be safe and fast, we can loop over heights if n_q is small, but 2000 is okay for vectorized ops if we use bincount or add.at.
    
    # Let's try to vectorize the scatter.
    # idx_plus = idx[:, 1:]  # shape (n_q, 63)
    # idx_minus = idx[:, :-1] # shape (n_q, 63)
    
    idx_plus = idx[:, 1:]
    idx_minus = idx[:, :-1]
    
    # Flatten
    active_flat = active.ravel()
    idx_plus_flat = idx_plus.ravel()
    idx_minus_flat = idx_minus.ravel()
    
    # We only care about active ones
    active_indices = np.where(active_flat)[0]
    
    if len(active_indices) > 0:
        idx_p_active = idx_plus_flat[active_indices]
        idx_m_active = idx_minus_flat[active_indices]
        
        # Contributions: +a[idx_p] to grad[idx_p], -a[idx_m] to grad[idx_m]
        # a is shape (64,)
        # We need to map idx to a[idx]
        
        # Use np.add.at
        np.add.at(grad, idx_p_active, a[idx_p_active])
        np.add.at(grad, idx_m_active, -a[idx_m_active])
        
    # The area is the mean of union_lengths over n_q points.
    # So the gradient is also divided by n_q.
    grad /= n_q
    
    return area, grad

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    Uses L-BFGS-B with analytical gradients.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    n = 64
    
    # Initial Keich construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0 = np.array(positions, dtype=np.float64)
    
    best_x = x0.copy()
    # Use high precision area for initial best evaluation? 
    # No, just use the fast one for consistency during search, 
    # but we want to minimize the TRUE area (20000 points).
    # The fast area (2000 points) is a good proxy.
    
    def objective(x):
        area, grad = _compute_area_and_grad(x, n_q=2000)
        return area, grad

    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 1000,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxfun': 5000,
                'maxls': 50
            }
        )
        
        if result.fun < _compute_area_and_grad(best_x, n_q=2000)[0]:
            best_x = result.x.copy()
            
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 60:
        return best_x

    # Second round from the best point, maybe with more iterations or different params
    try:
        result2 = minimize(
            objective,
            best_x,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 1000,
                'ftol': 1e-14,
                'gtol': 1e-10,
                'maxfun': 5000,
                'maxls': 50
            }
        )
        
        if result2.fun < _compute_area_and_grad(best_x, n_q=2000)[0]:
            best_x = result2.x.copy()
            
    except Exception:
        pass

    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return best_x

    # Final polish with higher resolution if time allows?
    # The evaluator uses 20000 points.
    # We can do a few steps with 20000 points to align with the metric exactly.
    
    def objective_high_res(x):
        area, grad = _compute_area_and_grad(x, n_q=20000)
        return area, grad

    try:
        result3 = minimize(
            objective_high_res,
            best_x,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 50,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxfun': 100,
                'maxls': 50
            }
        )
        
        if result3.fun < _compute_area_and_grad(best_x, n_q=20000)[0]:
            best_x = result3.x.copy()
            
    except Exception:
        pass

    return best_x

# EVOLVE-BLOCK-END
