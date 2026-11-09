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


def _soft_min_area(x, M=2000, beta=100.0):
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
    
    # soft_min(a,b) = min(a,b) - (1/beta)*log(1 + exp(-beta*|a-b|))
    diff_abs = np.abs(diffs - w_col)
    s_min = np.minimum(diffs, w_col)
    
    exp_term = np.exp(-beta * diff_abs)
    soft_correction = (1.0 / beta) * np.log1p(exp_term)
    soft_min_vals = s_min - soft_correction
    
    union_len = w + np.sum(soft_min_vals, axis=1)
    area = np.mean(union_len)
    
    # Gradient: d(soft_min(a,b))/da = sigmoid(-beta*(a-b))
    d_diff = diffs - w_col
    g_plus = 1.0 / (1.0 + np.exp(-beta * d_diff))
    
    grad = np.zeros(n, dtype=np.float64)
    
    g_flat = g_plus.flatten()
    order_plus_flat = order[:, 1:].flatten()
    order_minus_flat = order[:, :-1].flatten()
    
    np.add.at(grad, order_plus_flat, g_flat)
    np.add.at(grad, order_minus_flat, -g_flat)
    
    grad /= M
    
    return area, grad


def _union_area_grad(x, M=20000):
    """
    Compute the union area and its gradient w.r.t. x using exact min.
    
    x: shape (64,) - base positions
    M: number of quadrature points
    Returns: (area, grad) where grad has shape (64,)
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    
    order = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, order, axis=1)
    
    diffs = np.diff(L_sorted, axis=1)
    
    mask = diffs < w[:, np.newaxis]
    
    union_len = w + np.sum(np.minimum(diffs, w[:, np.newaxis]), axis=1)
    area = np.mean(union_len)
    
    grad = np.zeros(n, dtype=np.float64)
    
    rows, cols = np.where(mask)
    
    idx_plus = order[rows, cols]
    idx_minus = order[rows, cols + 1]
    
    np.add.at(grad, idx_plus, 1.0)
    np.add.at(grad, idx_minus, -1.0)
    
    grad /= M
    
    return area, grad


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


def _lbfgs_optimize_soft(x0, maxiter=200, M=2000, beta=100.0):
    """
    Run L-BFGS-B optimization using the soft-min surrogate.
    Returns (x_opt, soft_area_opt, exact_area_opt).
    """
    from scipy.optimize import minimize
    
    def objective_soft(x):
        area, grad = _soft_min_area(x, M=M, beta=beta)
        return area, grad
    
    res = minimize(
        objective_soft,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'gtol': 1e-9, 'ftol': 1e-12}
    )
    
    x_opt = res.x
    soft_area_opt = _soft_min_area(x_opt, M=M, beta=beta)[0]
    exact_area_opt = _union_area(x_opt, M=M)
    
    return x_opt, soft_area_opt, exact_area_opt


def _lbfgs_optimize_exact(x0, maxiter=200, M=2000):
    """
    Run L-BFGS-B optimization using the exact objective.
    Returns (x_opt, area_opt).
    """
    from scipy.optimize import minimize
    
    def objective_exact(x):
        area, grad = _union_area_grad(x, M=M)
        return area, grad
    
    res = minimize(
        objective_exact,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'gtol': 1e-9, 'ftol': 1e-12}
    )
    
    x_opt = res.x
    area_opt = _union_area(x_opt, M=M)
    
    return x_opt, area_opt


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Uses basin hopping with soft-min surrogate for smooth optimization,
    followed by exact polishing. Multiple diverse initial points and
    adaptive perturbation help escape local minima.
    
    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    
    start_time = time.time()
    deadline = start_time + 620.0
    
    def time_left():
        return deadline - time.time()
    
    best_x = _keich_init(n)
    best_area = _union_area(best_x, M=2000)
    
    # Phase 1: Generate diverse initial points and run soft-min optimization
    inits = []
    
    # Keich construction
    inits.append(_keich_init(n))
    
    # Perturbed Keich with various noise levels
    for seed in range(12):
        rng = np.random.default_rng(seed)
        noise_level = 0.02 + 0.03 * (seed % 4)
        inits.append(_keich_init(n) + rng.normal(0, noise_level, size=n))
    
    # Random initializations
    for seed in range(6):
        inits.append(_random_init(n, seed=200 + seed))
    
    # Sort inits by quick area estimate
    init_areas = [_union_area(x, M=500) for x in inits]
    init_order = np.argsort(init_areas)
    inits = [inits[i] for i in init_order]
    
    # Phase 1: Soft-min optimization from each initial point
    for i, x0 in enumerate(inits):
        if time_left() < 30.0:
            break
        
        try:
            x_opt, soft_area, exact_area = _lbfgs_optimize_soft(x0, maxiter=200, M=2000, beta=100.0)
            
            if exact_area < best_area:
                best_x = x_opt.copy()
                best_area = exact_area
        except Exception:
            continue
    
    # Phase 2: Basin hopping - perturb best solution and re-optimize
    rng = np.random.default_rng(42)
    n_hops = 0
    max_hops = 80
    hop_size = 0.04
    
    for hop in range(max_hops):
        if time_left() < 60.0:
            break
        if (time.time() - start_time) > 450.0:
            break
        
        perturbation = rng.normal(0, hop_size, size=n)
        x_perturbed = best_x + perturbation
        
        try:
            x_opt, soft_area, exact_area = _lbfgs_optimize_soft(x_perturbed, maxiter=150, M=2000, beta=100.0)
            
            if exact_area < best_area:
                best_x = x_opt.copy()
                best_area = exact_area
                hop_size = max(0.01, hop_size * 0.85)
            else:
                hop_size = min(0.15, hop_size * 1.15)
        except Exception:
            continue
    
    # Phase 3: Final polish with exact objective at full resolution
    if time_left() > 15.0:
        try:
            x_final, area_final = _lbfgs_optimize_exact(best_x, maxiter=80, M=20000)
            if area_final < best_area:
                best_x = x_final.copy()
                best_area = area_final
        except Exception:
            pass
    
    # Phase 4: Very tight final polish if time allows
    if time_left() > 10.0:
        try:
            x_final, area_final = _lbfgs_optimize_exact(best_x, maxiter=50, M=20000)
            if area_final < best_area:
                best_x = x_final.copy()
                best_area = area_final
        except Exception:
            pass
    
    # Phase 5: One more pass with slightly different initial perturbations
    if time_left() > 20.0:
        rng2 = np.random.default_rng(123)
        for _ in range(5):
            if time_left() < 15.0:
                break
            perturbation = rng2.normal(0, 0.03, size=n)
            x_perturbed = best_x + perturbation
            try:
                x_opt, soft_area, exact_area = _lbfgs_optimize_soft(x_perturbed, maxiter=100, M=2000, beta=150.0)
                if exact_area < best_area:
                    best_x = x_opt.copy()
                    best_area = exact_area
            except Exception:
                continue
        # Final exact polish
        if time_left() > 10.0:
            try:
                x_final, area_final = _lbfgs_optimize_exact(best_x, maxiter=40, M=20000)
                if area_final < best_area:
                    best_x = x_final.copy()
                    best_area = area_final
            except Exception:
                pass
    
    return best_x


# EVOLVE-BLOCK-END
