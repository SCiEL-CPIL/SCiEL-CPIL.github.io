# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_area_and_grad(x, n=64, M=20000):
    """Compute area and gradient of union area using vectorized operations."""
    a = (np.arange(n) + 1) / n
    w0 = 1.0 / n
    y = (np.arange(M) + 0.5) / M
    
    # L_i(y) = x_i + a_i * y, shape (n, M)
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) * w0
    
    # Sort L at each y
    idx = np.argsort(L, axis=0)
    L_sorted = np.take_along_axis(L, idx, axis=0)
    
    # Gaps between consecutive sorted L values
    gaps = np.diff(L_sorted, axis=0)
    
    # Union length = w + sum(min(gap, w))
    union_lengths = w + np.minimum(gaps, w[None, :]).sum(axis=0)
    area = np.mean(union_lengths)
    
    # Gradient: d(area)/d(x_i) using bincount
    active = gaps < w[None, :]
    
    left_idx = idx[:-1, :].ravel()
    right_idx = idx[1:, :].ravel()
    active_flat = active.ravel()
    
    left_counts = np.bincount(left_idx, weights=active_flat, minlength=n)
    right_counts = np.bincount(right_idx, weights=active_flat, minlength=n)
    
    grad_x = (left_counts - right_counts) / M
    
    return area, grad_x

def _compute_area_softmin_grad(x, n=64, M=20000, beta=50.0):
    """
    Compute area and gradient using soft-min surrogate.
    """
    a = (np.arange(n) + 1) / n
    w0 = 1.0 / n
    y = (np.arange(M) + 0.5) / M
    
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) * w0
    
    idx = np.argsort(L, axis=0)
    L_sorted = np.take_along_axis(L, idx, axis=0)
    gaps = np.diff(L_sorted, axis=0)
    
    diff = gaps - w[None, :]
    abs_diff = np.abs(diff)
    
    # Stable soft-min calculation
    # softmin(a,b) = min(a,b) - beta * log(1 + exp(-|a-b|/beta))
    soft_correction = beta * np.log1p(np.exp(-abs_diff / beta))
    soft_min_vals = np.minimum(gaps, w[None, :]) - soft_correction
    
    union_lengths = w + soft_min_vals.sum(axis=0)
    area = np.mean(union_lengths)
    
    # Gradient of soft-min w.r.t. gap is sigmoid(diff/beta)
    sigmoid_vals = 1.0 / (1.0 + np.exp(-diff / beta))
    
    left_idx = idx[:-1, :].ravel()
    right_idx = idx[1:, :].ravel()
    sig_flat = sigmoid_vals.ravel()
    
    left_counts = np.bincount(left_idx, weights=sig_flat, minlength=n)
    right_counts = np.bincount(right_idx, weights=sig_flat, minlength=n)
    
    grad_x = (left_counts - right_counts) / M
    
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

def _coordinate_descent_polish(x, n=64, M=20000, max_rounds=50, deadline=None):
    """
    Coordinate descent polish: for each coordinate, use binary search to find
    the optimal position that minimizes area.
    """
    x = x.copy()
    best_area = _compute_area_fast(x, n, M)
    a = (np.arange(n) + 1) / n
    w0 = 1.0 / n
    y = (np.arange(M) + 0.5) / M
    
    L_base = a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) * w0
    
    def _area_with_xi(xi_val, xi_idx):
        L = x[:, None] + L_base
        L[xi_idx, :] = xi_val + L_base[xi_idx, :]
        idx = np.argsort(L, axis=0)
        L_sorted = np.take_along_axis(L, idx, axis=0)
        gaps = np.diff(L_sorted, axis=0)
        union_lengths = w + np.minimum(gaps, w[None, :]).sum(axis=0)
        return np.mean(union_lengths)
    
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]
    
    for round_idx in range(max_rounds):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline - 10:
                break
            
            current_xi = x[i]
            best_xi = current_xi
            best_local_area = best_area
            
            for step in step_sizes:
                for direction in [1, -1]:
                    new_xi = current_xi + direction * step
                    try:
                        new_area = _area_with_xi(new_xi, i)
                        if new_area < best_local_area - 1e-14:
                            best_local_area = new_area
                            best_xi = new_xi
                    except Exception:
                        pass
            
            if best_xi != current_xi:
                if best_xi > current_xi:
                    lo = current_xi
                    hi = best_xi + step_sizes[0]
                else:
                    lo = best_xi - step_sizes[0]
                    hi = current_xi
                
                for _ in range(20):
                    mid = (lo + hi) / 2
                    area_mid = _area_with_xi(mid, i)
                    area_lo = _area_with_xi(lo, i)
                    area_hi = _area_with_xi(hi, i)
                    
                    if area_mid <= area_lo and area_mid <= area_hi:
                        lo = mid - (hi - lo) / 4
                        hi = mid + (hi - lo) / 4
                    elif area_lo < area_mid and area_lo < area_hi:
                        hi = mid
                    else:
                        lo = mid
                    
                    if area_mid < best_local_area - 1e-14:
                        best_local_area = area_mid
                        best_xi = mid
                
                x[i] = best_xi
                best_area = best_local_area
                improved = True
        
        if not improved:
            break
    
    return x, best_area

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
    
    Uses multi-restart L-BFGS-B optimization with low-resolution quadrature for speed,
    a soft-min smooth surrogate for navigating nonsmooth regions, and
    coordinate descent polish with high-resolution quadrature for precision.
    """
    start_time = time.time()
    deadline = start_time + 625.0
    
    n = 64
    M_low = 4000   # Low resolution for fast optimization
    M_high = 20000 # High resolution for final evaluation and polishing
    
    # Generate diverse initial points
    x_keich = _keich_initial(n)
    inits = [x_keich.copy()]
    
    rng = np.random.default_rng(42)
    for trial in range(12):
        scale = 0.005 * (trial + 1)
        x_pert = x_keich + rng.normal(0, scale, size=n)
        inits.append(x_pert)
    
    for trial in range(8):
        x_rand = rng.uniform(-0.5, 0.5, size=n)
        inits.append(x_rand)
    
    best_x = x_keich.copy()
    best_area = _compute_area_fast(x_keich, n, M_high)
    
    for init_idx, x0 in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:  # Leave 60s margin for final polish
            break
        
        # Phase 1: L-BFGS-B with exact nonsmooth objective (Low Res)
        try:
            def fun_and_jac_low(x_eval):
                area, grad = _compute_area_and_grad(x_eval, n, M_low)
                return area, grad
            
            result = minimize(
                fun=fun_and_jac_low,
                x0=x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            # Evaluate with High Res
            area_high = _compute_area_fast(result.x, n, M_high)
            if area_high < best_area:
                best_area = area_high
                best_x = result.x.copy()
        except Exception:
            pass
        
        # Phase 2: L-BFGS-B with soft-min smooth surrogate (Low Res)
        elapsed = time.time() - start_time
        if elapsed < deadline - 60:
            try:
                def fun_and_jac_soft_low(x_eval):
                    area, grad = _compute_area_softmin_grad(x_eval, n, M_low, beta=30.0)
                    return area, grad
                
                result_soft = minimize(
                    fun=fun_and_jac_soft_low,
                    x0=best_x,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                
                # Evaluate with High Res
                area_high = _compute_area_fast(result_soft.x, n, M_high)
                if area_high < best_area:
                    best_area = area_high
                    best_x = result_soft.x.copy()
            except Exception:
                pass
    
    # Final Polish: L-BFGS-B with High Res
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        try:
            def fun_and_jac_high(x_eval):
                area, grad = _compute_area_and_grad(x_eval, n, M_high)
                return area, grad
            
            result = minimize(
                fun=fun_and_jac_high,
                x0=best_x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.fun < best_area:
                area_full = _compute_area_fast(result.x, n, M_high)
                if area_full < best_area:
                    best_area = area_full
                    best_x = result.x.copy()
        except Exception:
            pass
    
    # Coordinate Descent Polish with High Res
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        try:
            polished_x, polished_area = _coordinate_descent_polish(
                best_x, n, M_high, max_rounds=50, deadline=deadline
            )
            if polished_area < best_area:
                best_area = polished_area
                best_x = polished_x.copy()
        except Exception:
            pass
    
    # Ensure valid output
    if not np.all(np.isfinite(best_x)):
        best_x = x_keich
    
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
