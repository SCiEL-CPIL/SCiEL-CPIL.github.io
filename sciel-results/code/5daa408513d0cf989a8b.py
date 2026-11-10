# EVOLVE-BLOCK-START
import numpy as np
import time


def _union_area(x, M=20000):
    """
    Compute the union area of 64 triangles.
    
    x: shape (64,) - base positions
    M: number of quadrature points
    Returns: area (scalar)
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    
    # L_i(y) = x_i + a_i * y
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    
    # Sort L along axis=1
    order = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, order, axis=1)
    
    # Consecutive differences
    diffs = np.diff(L_sorted, axis=1)
    
    # union_len at each y
    union_len = w + np.sum(np.minimum(diffs, w[:, np.newaxis]), axis=1)
    area = np.mean(union_len)
    
    return area


def _union_area_grad(x, M=20000):
    """
    Compute the union area and its gradient w.r.t. x using exact min.
    Uses bincount for fast gradient accumulation.
    
    x: shape (64,) - base positions
    M: number of quadrature points
    Returns: (area, grad) where grad has shape (64,)
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    
    # L_i(y) = x_i + a_i * y, shape (M, n)
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    
    # Sort L along axis=1 and track permutation
    order = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, order, axis=1)
    
    # Consecutive differences
    diffs = np.diff(L_sorted, axis=1)
    
    # mask: (M, n-1) boolean, True where diffs < w (overlapping)
    mask = diffs < w[:, np.newaxis]
    
    # union_len = w + sum min(diffs, w)
    union_len = w + np.sum(np.minimum(diffs, w[:, np.newaxis]), axis=1)
    area = np.mean(union_len)
    
    # Gradient: where mask is True, +1 to left index, -1 to right index
    m_idx, j_idx = np.where(mask)
    
    if len(m_idx) > 0:
        left_indices = order[m_idx, j_idx]
        right_indices = order[m_idx, j_idx + 1]
        
        grad_right = np.bincount(right_indices, minlength=n).astype(np.float64)
        grad_left = np.bincount(left_indices, minlength=n).astype(np.float64)
        grad = grad_right - grad_left
    else:
        grad = np.zeros(n, dtype=np.float64)
    
    grad /= M
    
    return area, grad


def _soft_min_area(x, M=2000, beta=50.0):
    """
    Compute a smooth surrogate for the union area using soft-min.
    
    x: shape (64,) - base positions
    M: number of quadrature points
    beta: inverse temperature for soft-min (higher = sharper)
    Returns: (area, grad) where grad has shape (64,)
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    
    # L_i(y) = x_i + a_i * y, shape (M, n)
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    
    # Sort L along axis=1 and track permutation
    order = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, order, axis=1)
    
    # Consecutive differences
    diffs = np.diff(L_sorted, axis=1)  # shape (M, n-1)
    w_col = w[:, np.newaxis]  # shape (M, 1)
    
    # Soft-min of (diffs, w):
    # soft_min(a,b) = min(a,b) - (1/beta)*log(1 + exp(-beta*|a-b|))
    diff_abs = np.abs(diffs - w_col)
    s_min = np.minimum(diffs, w_col)
    
    # Stable computation: -beta*|d| <= 0, so exp is safe
    exp_term = np.exp(-beta * diff_abs)
    soft_correction = (1.0 / beta) * np.log1p(exp_term)
    soft_min_vals = s_min - soft_correction  # shape (M, n-1)
    
    # union_len = w + sum soft_min_vals
    union_len = w + np.sum(soft_min_vals, axis=1)
    area = np.mean(union_len)
    
    # Gradient: d(soft_min)/d(diffs) = sigmoid(-beta*(diffs-w))
    d_diff = diffs - w_col
    # Stable sigmoid: sigmoid(z) = 1/(1+exp(-z)), use piecewise for stability
    g_plus = np.empty_like(d_diff)
    mask_pos = d_diff >= 0
    # For d_diff >= 0: sigmoid(-beta*d_diff) = exp(-beta*d_diff)/(1+exp(-beta*d_diff))
    exp_neg = np.exp(-beta * d_diff[mask_pos])
    g_plus[mask_pos] = exp_neg / (1.0 + exp_neg)
    # For d_diff < 0: sigmoid(-beta*d_diff) = 1/(1+exp(beta*d_diff))
    exp_pos = np.exp(beta * d_diff[~mask_pos])
    g_plus[~mask_pos] = 1.0 / (1.0 + exp_pos)
    
    # Accumulate gradient using bincount for speed
    # g_plus has shape (M, n-1)
    # For each (m,k): grad[order[m,k+1]] += g_plus[m,k], grad[order[m,k]] -= g_plus[m,k]
    
    # Flatten
    g_flat = g_plus.flatten()
    order_plus_flat = order[:, 1:].flatten()
    order_minus_flat = order[:, :-1].flatten()
    
    # Use bincount with weights for fast accumulation
    grad_plus = np.bincount(order_plus_flat, weights=g_flat, minlength=n)
    grad_minus = np.bincount(order_minus_flat, weights=g_flat, minlength=n)
    grad = grad_plus - grad_minus
    
    # Scale by 1/M (mean)
    grad /= M
    
    return area, grad


def _subgradient_polish(x, M=20000, max_iter=50, timeout=20.0):
    """
    Subgradient method polish for nonsmooth objective.
    Handles the kinks in the min(diffs, w) term directly.
    
    x: shape (64,) - base positions
    M: number of quadrature points
    max_iter: maximum number of iterations
    timeout: time limit in seconds
    Returns: (x_opt, area_opt)
    """
    n = 64
    start = time.time()
    
    # Adaptive step size schedule
    step_sizes = np.linspace(0.05, 0.001, max_iter)
    
    best_x = x.copy()
    best_area = _union_area(x, M=M)
    
    for it in range(max_iter):
        if time.time() - start > timeout:
            break
        
        area, grad = _union_area_grad(x, M=M)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        # Subgradient step with adaptive size
        step = step_sizes[min(it, max_iter - 1)]
        x_new = x - step * grad / (grad_norm + 1e-8)
        
        area_new = _union_area(x_new, M=M)
        
        if area_new < best_area:
            best_x = x_new.copy()
            best_area = area_new
            x = x_new
        else:
            # Reduce step size on failure
            x = x_new * 0.5 + x * 0.5  # Damped step
            area_check = _union_area(x, M=M)
            if area_check < best_area:
                best_x = x.copy()
                best_area = area_check
    
    return best_x, best_area


def _keich_init(n=64):
    """
    Keich construction initial guess.
    """
    k = int(np.log2(n))
    j_arr = np.arange(1, k + 1)
    positions = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)
    return positions


def _random_init(n=64, seed=0):
    """
    Random initial guess.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, size=n)


def _nm_optimize(x0, M=2000, maxiter=200, timeout=30.0):
    """
    Nelder-Mead optimization on the exact objective.
    Robust to non-smoothness.
    """
    from scipy.optimize import minimize
    
    start = time.time()
    
    def objective(x):
        if time.time() - start > timeout:
            raise TimeoutError("NM timeout")
        return _union_area(x, M=M)
    
    try:
        res = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        return res.x
    except (TimeoutError, Exception):
        return x0


def _lbfgs_optimize_soft(x0, maxiter=200, M=2000, beta_start=50.0, beta_end=400.0, timeout=30.0):
    """
    Run L-BFGS-B optimization using the soft-min surrogate with increasing beta schedule.
    Beta increases from beta_start to beta_end over the optimization, progressively
    sharpening toward the true nonsmooth objective.
    Returns (x_opt, exact_area_opt).
    """
    from scipy.optimize import minimize
    
    start = time.time()
    current_beta = [beta_start]
    total_iterations = [0]
    
    def objective_soft(x):
        if time.time() - start > timeout:
            raise TimeoutError("L-BFGS timeout")
        
        # Increase beta over iterations (exponential schedule)
        progress = min(total_iterations[0] / max(maxiter, 1), 1.0)
        # Exponential interpolation: beta = beta_start * (beta_end/beta_start)^progress
        current_beta[0] = beta_start * (beta_end / beta_start) ** progress
        total_iterations[0] += 1
        
        area, grad = _soft_min_area(x, M=M, beta=current_beta[0])
        return area, grad
    
    try:
        res = minimize(
            objective_soft,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-10}
        )
        x_opt = res.x
        exact_area_opt = _union_area(x_opt, M=M)
        return x_opt, exact_area_opt
    except (TimeoutError, Exception):
        return x0, _union_area(x0, M=M)


def _lbfgs_optimize_exact(x0, maxiter=50, M=20000, timeout=30.0):
    """
    Run L-BFGS-B optimization using the exact objective.
    Returns (x_opt, area_opt).
    """
    from scipy.optimize import minimize
    
    start = time.time()
    
    def objective_exact(x):
        if time.time() - start > timeout:
            raise TimeoutError("L-BFGS exact timeout")
        area, grad = _union_area_grad(x, M=M)
        return area, grad
    
    try:
        res = minimize(
            objective_exact,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': 1e-10, 'ftol': 1e-12}
        )
        x_opt = res.x
        area_opt = _union_area(x_opt, M=M)
        return x_opt, area_opt
    except (TimeoutError, Exception):
        return x0, _union_area(x0, M=M)


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Strategy:
    1. Generate diverse initial points (Keich + perturbations + random)
    2. Optimize each with soft-min L-BFGS (smooth, fast)
    3. Basin hopping with adaptive perturbation
    4. Final polish: Nelder-Mead on exact objective (robust to non-smoothness)
    5. Tight L-BFGS polish at full resolution
    
    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    
    start_time = time.time()
    deadline = start_time + 620.0  # 620s limit (10s safety margin for 630s kill)
    
    def time_left():
        return deadline - time.time()
    
    # Initialize with Keich
    best_x = _keich_init(n)
    best_area = _union_area(best_x, M=2000)
    
    # Phase 1: Generate diverse initial points
    inits = []
    
    # Keich construction
    inits.append(_keich_init(n))
    
    # Perturbed Keich with various noise levels
    for seed in range(10):
        rng = np.random.default_rng(seed)
        noise_level = 0.02 + 0.05 * (seed % 4)
        inits.append(_keich_init(n) + rng.normal(0, noise_level, size=n))
    
    # Random initializations
    for seed in range(5):
        inits.append(_random_init(n, seed=200 + seed))
    
    # Sort inits by quick area estimate
    init_areas = [_union_area(x, M=500) for x in inits]
    init_order = np.argsort(init_areas)
    inits = [inits[i] for i in init_order]
    
    # Phase 2: Soft-min L-BFGS optimization from each initial point with beta schedule
    for i, x0 in enumerate(inits):
        if time_left() < 60.0:
            break
        
        try:
            x_opt, exact_area = _lbfgs_optimize_soft(x0, maxiter=150, M=2000, beta_start=50.0, beta_end=400.0, timeout=40.0)
            
            if exact_area < best_area:
                best_x = x_opt.copy()
                best_area = exact_area
        except Exception:
            continue
    
    # Phase 3: Basin hopping - perturb best solution and re-optimize
    rng = np.random.default_rng(42)
    max_hops = 50
    hop_size = 0.05
    
    for hop in range(max_hops):
        if time_left() < 80.0:
            break
        if time.time() - start_time > 480.0:  # Reserve time for final polish
            break
        
        # Generate perturbation with occasional larger jumps
        if hop % 10 == 9:
            current_hop = hop_size * 2.0  # Larger jump every 10th iteration
        else:
            current_hop = hop_size
        
        perturbation = rng.normal(0, current_hop, size=n)
        x_perturbed = best_x + perturbation
        
        try:
            # Optimize from perturbed point using soft-min with beta schedule
            x_opt, exact_area = _lbfgs_optimize_soft(x_perturbed, maxiter=100, M=2000, beta_start=50.0, beta_end=400.0, timeout=30.0)
            
            if exact_area < best_area:
                best_x = x_opt.copy()
                best_area = exact_area
                hop_size = max(0.01, hop_size * 0.8)  # Reduce hop size on success
            else:
                hop_size = min(0.2, hop_size * 1.15)  # Increase hop size on failure
        except Exception:
            continue
    
    # Phase 4: Subgradient polish on exact objective (handles nonsmoothness directly)
    if time_left() > 80.0:
        try:
            x_sg, area_sg = _subgradient_polish(best_x, M=20000, max_iter=30, timeout=40.0)
            if area_sg < best_area:
                best_x = x_sg.copy()
                best_area = area_sg
        except Exception:
            pass
    
    # Phase 5: Nelder-Mead polish on exact objective (handles non-smoothness well)
    if time_left() > 50.0:
        try:
            x_nm = _nm_optimize(best_x, M=4000, maxiter=200, timeout=30.0)
            area_nm = _union_area(x_nm, M=4000)
            if area_nm < best_area:
                best_x = x_nm.copy()
                best_area = area_nm
        except Exception:
            pass
    
    # Phase 6: L-BFGS polish with exact objective at full resolution
    if time_left() > 40.0:
        try:
            x_final, area_final = _lbfgs_optimize_exact(best_x, maxiter=50, M=20000, timeout=30.0)
            if area_final < best_area:
                best_x = x_final.copy()
                best_area = area_final
        except Exception:
            pass
    
    # Phase 7: Final subgradient polish at full resolution if time allows
    if time_left() > 20.0:
        try:
            x_final, area_final = _subgradient_polish(best_x, M=20000, max_iter=20, timeout=15.0)
            if area_final < best_area:
                best_x = x_final.copy()
                best_area = area_final
        except Exception:
            pass
    
    # Phase 8: Final tight Nelder-Mead at full resolution if time allows
    if time_left() > 15.0:
        try:
            x_final = _nm_optimize(best_x, M=20000, maxiter=50, timeout=10.0)
            area_final = _union_area(x_final, M=20000)
            if area_final < best_area:
                best_x = x_final.copy()
                best_area = area_final
        except Exception:
            pass
    
    # Final validation
    final_area = _union_area(best_x, M=20000)
    if not np.isfinite(final_area) or final_area <= 0:
        # Fallback to Keich
        best_x = _keich_init(n)
    
    return best_x


# EVOLVE-BLOCK-END
