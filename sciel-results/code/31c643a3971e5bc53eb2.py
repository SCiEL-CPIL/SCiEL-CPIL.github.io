# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_P_fast(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] using the fast CDF method.
    support: 1-D array of non-negative floats
    weights: 1-D array of non-negative floats (will be normalized)
    Returns: float P
    """
    n = len(support)
    if n == 0:
        return 0.0
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    p = weights / w_sum
    s = support

    # Distribution of S2 = X1 + X2
    s_i = s[:, None]
    s_j = s[None, :]
    sums_2 = s_i + s_j
    prods_2 = p[:, None] * p[None, :]

    sums_2_flat = sums_2.flatten()
    prods_2_flat = prods_2.flatten()

    unique_sums_2, inverse_indices_2 = np.unique(sums_2_flat, return_inverse=True)
    probs_2 = np.bincount(inverse_indices_2, weights=prods_2_flat)

    sort_idx = np.argsort(unique_sums_2)
    sorted_sums_2 = unique_sums_2[sort_idx]
    sorted_probs_2 = probs_2[sort_idx]
    cdf_2 = np.cumsum(sorted_probs_2)

    # P = sum_{k, l} p_k * p_l * P(S2 < 2*s_l - s_k)
    # T[k,l] = 2*s_l - s_k
    T = 2.0 * s[None, :] - s[:, None]  # shape (n, n), T[k, l] = 2*s[l] - s[k]

    # For each (k, l), find number of sorted_sums_2 < T[k, l]
    idx_arr = np.searchsorted(sorted_sums_2, T, side='left')
    # p_cond[k, l] = cdf_2[idx_arr[k,l] - 1] if idx_arr[k,l] > 0 else 0
    p_cond = np.where(idx_arr > 0, cdf_2[np.maximum(idx_arr - 1, 0)], 0.0)

    P_total = np.sum(p[:, None] * p[None, :] * p_cond)

    return float(P_total)


def _neg_P_from_params(params, n):
    """Compute -P from parameter vector: first n-1 are supports (s[0]=0 fixed), next n are weights."""
    s = np.zeros(n, dtype=np.float64)
    s[1:] = params[:n-1]
    w = params[n-1:]
    
    # Ensure validity
    if np.any(s < -1e-9) or np.any(w < -1e-9):
        return 1e10
    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return 1e10
    
    p = w / w_sum
    P = _compute_P_fast(s, p)
    if not np.isfinite(P):
        return 1e10
    return -P


def _optimize_weights_for_support(support, start_weights=None, maxiter=1000):
    """
    Optimize weights for a fixed support using Nelder-Mead.
    Returns (best_weights, best_P)
    """
    n = len(support)
    
    if start_weights is None:
        start_weights = np.ones(n) / n
    
    def _neg_P_weights(w_params):
        w = np.maximum(w_params, 0.0)
        w_sum = np.sum(w)
        if w_sum <= 1e-9:
            return 1e10
        w_norm = w / w_sum
        P = _compute_P_fast(support, w_norm)
        if not np.isfinite(P):
            return 1e10
        return -P
    
    try:
        result = minimize(
            _neg_P_weights,
            start_weights.copy(),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        
        w_opt = np.maximum(result.x, 0.0)
        w_sum = np.sum(w_opt)
        if w_sum <= 1e-9:
            return start_weights, 0.0
        w_opt = w_opt / w_sum
        P_opt = _compute_P_fast(support, w_opt)
        if not np.isfinite(P_opt):
            return start_weights, 0.0
        return w_opt, P_opt
    except Exception:
        return start_weights, 0.0


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = start_time + 620.0
    
    best_support = np.array([0.0, 1.0], dtype=np.float64)
    best_weights = np.array([0.5, 0.5], dtype=np.float64)
    best_P = _compute_P_fast(best_support, best_weights)
    
    def update_best(sup, wts, P_val):
        nonlocal best_support, best_weights, best_P
        if P_val > best_P:
            best_support = sup.copy()
            best_weights = wts.copy()
            best_P = P_val
    
    # Phase 1: Try structured initial configurations
    # 1a. Two-point distributions
    for a in [0.5, 1.0, 2.0, 5.0, 10.0]:
        for p0 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            support = np.array([0.0, a])
            weights = np.array([p0, 1.0 - p0])
            P = _compute_P_fast(support, weights)
            update_best(support, weights, P)
    
    # 1b. Three-point distributions
    for a in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        for b in [1.0, 2.0, 3.0, 5.0]:
            if a >= b:
                continue
            for p0 in [0.1, 0.2, 0.3, 0.4]:
                for p1 in [0.1, 0.2, 0.3, 0.4]:
                    p2 = 1.0 - p0 - p1
                    if p2 < 0.01:
                        continue
                    support = np.array([0.0, a, b])
                    weights = np.array([p0, p1, p2])
                    P = _compute_P_fast(support, weights)
                    update_best(support, weights, P)
    
    # 1c. Uniform on [0, 1] with various n
    for n in [8, 16, 32, 64]:
        support = np.linspace(0.0, 1.0, n)
        weights = np.ones(n) / n
        P = _compute_P_fast(support, weights)
        update_best(support, weights, P)
    
    # 1d. Beta-like distributions
    for n in [16, 32, 64]:
        support = np.linspace(0.0, 1.0, n)
        for alpha in [0.5, 1.0, 1.5, 2.0, 3.0]:
            for beta in [0.5, 1.0, 1.5, 2.0, 3.0]:
                x = np.clip(support, 1e-12, 1.0 - 1e-12)
                weights = (x ** (alpha - 1)) * ((1.0 - x) ** (beta - 1))
                w_sum = np.sum(weights)
                if w_sum <= 1e-9:
                    continue
                weights = weights / w_sum
                P = _compute_P_fast(support, weights)
                update_best(support, weights, P)
    
    # Phase 2: Joint optimization of support and weights using Nelder-Mead
    # s[0] is fixed at 0 by scale invariance
    n_values = [2, 3, 4, 5, 6, 7, 8]
    
    rng = np.random.default_rng(42)
    
    for n in n_values:
        if time.time() > deadline - 60:
            break
        
        # Number of free params: (n-1) supports + n weights = 2n-1
        # More restarts for smaller n, fewer for larger n
        n_restarts = max(3, int(100 / n))
        
        for restart in range(n_restarts):
            if time.time() > deadline - 30:
                break
            
            # Random initial guess
            if n == 2:
                s_init = np.array([0.0, rng.uniform(0.5, 10.0)])
            else:
                s_rest = np.sort(rng.uniform(0.5, 20.0, size=n-1))
                s_init = np.concatenate([[0.0], s_rest])
            
            w_init = rng.dirichlet(np.ones(n))
            
            # Build param vector: [s_1, ..., s_{n-1}, w_0, ..., w_{n-1}]
            params_init = np.concatenate([s_init[1:], w_init])
            
            # Bounds: supports > 0, weights > 0
            bounds = [(1e-6, 1e6)] * (n-1) + [(1e-6, 1.0)] * n
            
            try:
                result = minimize(
                    _neg_P_from_params,
                    params_init,
                    args=(n,),
                    method='Nelder-Mead',
                    bounds=bounds,
                    options={'maxiter': 800, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                )
                
                if result.fun > 1e9:  # invalid
                    continue
                
                P_val = -result.fun
                if not np.isfinite(P_val):
                    continue
                
                if P_val > best_P + 1e-12:
                    # Extract support and weights
                    s_opt = np.zeros(n, dtype=np.float64)
                    s_opt[1:] = result.x[:n-1]
                    w_opt = result.x[n-1:]
                    w_sum = np.sum(w_opt)
                    if w_sum > 1e-9:
                        w_opt = w_opt / w_sum
                        update_best(s_opt, w_opt, P_val)
            except Exception:
                continue
    
    # Phase 3: Refine best solution with weight-only optimization
    if time.time() < deadline - 20:
        w_refined, P_refined = _optimize_weights_for_support(
            best_support, best_weights, maxiter=2000
        )
        if P_refined > best_P:
            update_best(best_support, w_refined, P_refined)
    
    # Phase 4: Try a few more structured configurations near the best
    if time.time() < deadline - 10:
        # Perturb the best support and re-optimize weights
        for scale in [0.5, 1.5, 2.0]:
            s_pert = best_support * scale
            # Ensure s[0] = 0
            s_pert[0] = 0.0
            w_pert, P_pert = _optimize_weights_for_support(
                s_pert, best_weights, maxiter=500
            )
            if P_pert > best_P:
                update_best(s_pert, w_pert, P_pert)
    
    # Final validation
    if best_P <= 0:
        # Fallback
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([0.5, 0.5], dtype=np.float64)
    
    # Ensure all values are valid
    if np.any(best_support < -1e-9):
        best_support = np.maximum(best_support, 0.0)
    if np.any(best_weights < -1e-9):
        best_weights = np.maximum(best_weights, 0.0)
    w_sum = np.sum(best_weights)
    if w_sum <= 1e-9:
        best_weights = np.ones(len(best_support)) / len(best_support)
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
