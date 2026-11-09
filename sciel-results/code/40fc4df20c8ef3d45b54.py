# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _union_area(x, M=20000):
    """
    Compute the union area of 64 triangles.
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    
    order = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, order, axis=1)
    
    diffs = np.diff(L_sorted, axis=1)
    
    union_len = w + np.sum(np.minimum(diffs, w[:, np.newaxis]), axis=1)
    area = np.mean(union_len)
    
    return area


def _union_area_grad(x, M=20000):
    """
    Compute the union area and its gradient w.r.t. x.
    Uses bincount for fast gradient accumulation.
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
    Returns (area, grad) where grad has shape (64,).
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    
    order = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, order, axis=1)
    
    diffs = np.diff(L_sorted, axis=1)
    w_col = w[:, np.newaxis]
    
    # Soft-min: soft_min(a,b) = min(a,b) - (1/beta)*log(1 + exp(-beta*|a-b|))
    diff_abs = np.abs(diffs - w_col)
    s_min = np.minimum(diffs, w_col)
    
    exp_term = np.exp(-beta * diff_abs)
    soft_correction = (1.0 / beta) * np.log1p(exp_term)
    soft_min_vals = s_min - soft_correction
    
    union_len = w + np.sum(soft_min_vals, axis=1)
    area = np.mean(union_len)
    
    # Gradient w.r.t. diffs: sigmoid(-beta*(diffs - w))
    d_diff = diffs - w_col
    # stable sigmoid(-beta * d_diff)
    g_plus = np.empty_like(d_diff)
    mask_pos = d_diff >= 0
    g_plus[mask_pos] = 1.0 / (1.0 + np.exp(beta * d_diff[mask_pos]))
    exp_neg = np.exp(-beta * d_diff[~mask_pos])
    g_plus[~mask_pos] = exp_neg / (1.0 + exp_neg)
    
    g_flat = g_plus.flatten()
    order_plus_flat = order[:, 1:].flatten()
    order_minus_flat = order[:, :-1].flatten()
    
    grad_plus = np.bincount(order_plus_flat, weights=g_flat, minlength=n)
    grad_minus = np.bincount(order_minus_flat, weights=g_flat, minlength=n)
    grad = grad_plus - grad_minus
    
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


def _subgradient_polish(x, M=20000, max_iter=200, lr_init=0.01, deadline=None):
    """
    Projected subgradient method with adaptive step size for the exact nonsmooth objective.
    This respects the permutation-invariant structure by working directly on x.
    """
    n = 64
    best_x = x.copy()
    best_area = _union_area(best_x, M=M)
    
    lr = lr_init
    patience = 0
    max_patience = 30
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        area, grad = _union_area_grad(x, M=M)
        
        # Adaptive step size: decrease if no improvement, increase if stable
        x_new = x - lr * grad
        area_new = _union_area(x_new, M=M)
        
        if area_new < best_area:
            best_x = x_new.copy()
            best_area = area_new
            patience = 0
            # Slightly increase lr if improving consistently
            lr = min(lr * 1.05, 0.05)
        else:
            patience += 1
            lr *= 0.8
            x_new = x  # don't move
        
        x = x_new
        
        if patience > max_patience:
            break
        
        # Check convergence
        if np.linalg.norm(grad) < 1e-10:
            break
    
    return best_x, best_area


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Uses a multi-phase approach:
    1. Coarse optimization with soft-min smooth surrogate (increasing beta)
    2. Fine optimization with exact objective at higher resolution
    3. Subgradient polish with adaptive step sizes
    4. Full-resolution L-BFGS polish
    """
    n = 64
    
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin before 630s limit
    
    def time_left():
        return deadline - time.time()
    
    best_x = _keich_init(n)
    best_area = _union_area(best_x, M=2000)
    
    # Generate diverse initial points
    inits = []
    
    # Keich construction
    inits.append(_keich_init(n))
    
    # Perturbed Keich with different noise levels and seeds
    for seed in range(12):
        rng = np.random.default_rng(seed)
        noise_level = 0.01 + 0.05 * (seed % 6)
        inits.append(_keich_init(n) + rng.normal(0, noise_level, size=n))
    
    # Random initializations
    for seed in range(6):
        inits.append(_random_init(n, seed=200 + seed))
    
    # Structured initializations
    base = _keich_init(n)
    for shift in [-0.15, -0.08, 0.08, 0.15]:
        inits.append(base + shift)
    for scale in [0.7, 0.9, 1.1, 1.3]:
        inits.append(base * scale)
    
    # Sort inits by initial area (quick check)
    init_areas = [_union_area(x, M=800) for x in inits]
    init_order = np.argsort(init_areas)
    inits = [inits[i] for i in init_order]
    
    # Phase 1: Coarse optimization with soft-min surrogate, increasing beta
    coarse_M = 2000
    beta_schedule = [50.0, 100.0, 200.0, 400.0]
    
    for beta in beta_schedule:
        if time_left() < 30.0:
            break
        
        for i, x0 in enumerate(inits[:8]):  # Limit to top 8 inits for speed
            if time_left() < 20.0:
                break
            
            try:
                def objective_soft(x, beta=beta):
                    area, grad = _soft_min_area(x, M=coarse_M, beta=beta)
                    return area, grad
                
                res = minimize(
                    objective_soft,
                    x0,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'gtol': 1e-7, 'ftol': 1e-9, 'maxls': 30}
                )
                
                x_opt = res.x
                area_opt = _union_area(x_opt, M=coarse_M)
                
                if area_opt < best_area:
                    best_x = x_opt.copy()
                    best_area = area_opt
            except Exception:
                continue
        
        # Update inits with current best
        inits[0] = best_x.copy()
    
    # Phase 2: Fine optimization with exact objective at higher resolution
    fine_M = 5000
    
    for i in range(min(5, len(inits))):
        if time_left() < 25.0:
            break
        
        x0 = best_x + np.random.default_rng(i * 137 + 42).normal(0, 0.02, size=n)
        
        try:
            def objective_fine(x):
                area, grad = _union_area_grad(x, M=fine_M)
                return area, grad
            
            res = minimize(
                objective_fine,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 120, 'gtol': 1e-8, 'ftol': 1e-10, 'maxls': 40}
            )
            
            x_opt = res.x
            area_opt = _union_area(x_opt, M=fine_M)
            
            if area_opt < best_area:
                best_x = x_opt.copy()
                best_area = area_opt
        except Exception:
            continue
    
    # Phase 3: Subgradient polish with adaptive step sizes
    if time_left() > 60.0:
        try:
            x_sg, area_sg = _subgradient_polish(
                best_x, M=10000, max_iter=150, lr_init=0.005,
                deadline=start_time + 600.0
            )
            if area_sg < best_area:
                best_x = x_sg.copy()
                best_area = area_sg
        except Exception:
            pass
    
    # Phase 4: Full-resolution L-BFGS polish
    if time_left() > 30.0:
        try:
            def objective_full(x):
                area, grad = _union_area_grad(x, M=20000)
                return area, grad
            
            res = minimize(
                objective_full,
                best_x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 80, 'gtol': 1e-9, 'ftol': 1e-11, 'maxls': 50}
            )
            
            x_final = res.x
            area_final = _union_area(x_final, M=20000)
            
            if area_final < best_area:
                best_x = x_final.copy()
                best_area = area_final
        except Exception:
            pass
    
    # Phase 5: Second polish if time remains
    if time_left() > 15.0:
        try:
            def objective_full2(x):
                area, grad = _union_area_grad(x, M=20000)
                return area, grad
            
            res2 = minimize(
                objective_full2,
                best_x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 50, 'gtol': 1e-10, 'ftol': 1e-12, 'maxls': 50}
            )
            
            x_final2 = res2.x
            area_final2 = _union_area(x_final2, M=20000)
            
            if area_final2 < best_area:
                best_x = x_final2.copy()
                best_area = area_final2
        except Exception:
            pass
    
    # Ensure result is valid
    if not np.all(np.isfinite(best_x)):
        best_x = _keich_init(n)
    
    return best_x


# EVOLVE-BLOCK-END
