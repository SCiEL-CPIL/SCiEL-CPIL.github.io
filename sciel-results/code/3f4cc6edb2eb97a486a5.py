# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_area_and_grad(x, n=64, M=20000):
    """Compute area and gradient of union area using vectorized operations."""
    a = (np.arange(n) + 1) / n  # a_i = (i+1)/n
    w0 = 1.0 / n
    y = (np.arange(M) + 0.5) / M  # M quadrature points
    
    # L_i(y) = x_i + a_i * y, shape (n, M)
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) * w0  # width at each y, shape (M,)
    
    # Sort L at each y
    idx = np.argsort(L, axis=0)  # (n, M)
    L_sorted = np.take_along_axis(L, idx, axis=0)  # (n, M)
    
    # Gaps between consecutive sorted L values
    gaps = np.diff(L_sorted, axis=0)  # (n-1, M)
    
    # Union length = w + sum(min(gap, w))
    union_lengths = w + np.minimum(gaps, w[None, :]).sum(axis=0)  # (M,)
    area = np.mean(union_lengths)
    
    # Gradient computation using scatter operations
    # For each active gap (gap < w), the left endpoint gets +1, right gets -1
    active = gaps < w[None, :]  # (n-1, M)
    
    # Flatten for scatter operations
    left_idx = idx[:-1, :]  # (n-1, M)
    right_idx = idx[1:, :]  # (n-1, M)
    
    # Create flat arrays for scatter
    left_flat = left_idx.ravel()  # (n-1)*M
    right_flat = right_idx.ravel()  # (n-1)*M
    active_flat = active.ravel()  # (n-1)*M
    
    # Column index: entry j corresponds to m = j % M
    col_idx = np.arange((n-1) * M) % M  # (n-1)*M
    
    # Scatter-add: for left endpoints, add active_flat
    grad = np.zeros((n, M), dtype=np.float64)
    np.add.at(grad, (left_flat, col_idx), active_flat)
    # Scatter-add: for right endpoints, subtract active_flat
    np.add.at(grad, (right_flat, col_idx), -active_flat)
    
    # The gradient of area w.r.t. x_i is the mean of grad[i, :] over m
    grad_x = grad.mean(axis=1)  # (n,)
    
    return area, grad_x


def _compute_area_fast(x, n=64, M=20000):
    """Compute only the area (no gradient) for evaluation."""
    a = (np.arange(n) + 1) / n
    w0 = 1.0 / n
    y = (np.arange(M) + 0.5) / M
    
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) * w0
    
    idx = np.argsort(L, axis=0)
    L_sorted = np.take_along_axis(L, idx, axis=0)
    gaps = np.diff(L_sorted, axis=0)
    
    union_lengths = w + np.minimum(gaps, w[None, :]).sum(axis=0)
    area = np.mean(union_lengths)
    return area


def _keich_initial(n=64):
    """Generate the Keich construction initial point."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Uses multi-restart L-BFGS-B optimization with fast gradient computation
    to find base positions that minimize the union area of 64 triangles.
    """
    start_time = time.time()
    deadline = start_time + 580  # 580 second limit (20s safety margin)
    
    n = 64
    M = 20000  # Full quadrature resolution
    
    # Generate diverse initial points
    x_keich = _keich_initial(n)
    
    # Create multiple initializations
    inits = [x_keich.copy()]
    
    # Add perturbed versions of Keich
    rng = np.random.default_rng(42)
    for trial in range(10):
        scale = 0.01 * (trial + 1)
        x_pert = x_keich + rng.normal(0, scale, size=n)
        inits.append(x_pert)
    
    # Add a few random initializations
    for trial in range(5):
        x_rand = rng.uniform(-0.5, 0.5, size=n)
        inits.append(x_rand)
    
    # Evaluate each initialization and run optimization
    best_x = x_keich.copy()
    best_area = _compute_area_fast(x_keich, n, M)
    
    # State for combined fun+jac to avoid redundant computation
    _state = {'x': None, 'area': None, 'grad': None}
    
    def _fun_jac_combined(x):
        """Combined function and gradient evaluation."""
        # Check if we already computed this
        if _state['x'] is not None and np.array_equal(x, _state['x']):
            return _state['area'], _state['grad']
        
        area, grad = _compute_area_and_grad(x, n, M)
        _state['x'] = x.copy()
        _state['area'] = area
        _state['grad'] = grad
        return area, grad
    
    def _fun_only(x):
        area, _ = _fun_jac_combined(x)
        return area
    
    def _jac_only(x):
        _, grad = _fun_jac_combined(x)
        return grad
    
    for init_idx, x0 in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:  # Leave 60s margin
            break
        
        # Run L-BFGS-B with combined fun+jac
        try:
            result = minimize(
                fun=_fun_only,
                x0=x0,
                jac=_jac_only,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_area:
                # Evaluate at full resolution to confirm
                area_full = _compute_area_fast(result.x, n, M)
                if area_full < best_area:
                    best_area = area_full
                    best_x = result.x.copy()
        except Exception:
            continue
    
    # Final polish with more iterations if time permits
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        try:
            result = minimize(
                fun=_fun_only,
                x0=best_x,
                jac=_jac_only,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.fun < best_area:
                area_full = _compute_area_fast(result.x, n, M)
                if area_full < best_area:
                    best_area = area_full
                    best_x = result.x.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if not np.all(np.isfinite(best_x)):
        best_x = x_keich
    
    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
