# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as so
import time


def _compute_P_fast(support, weights):
    """Fast computation of P[X1+X2+X3 < 2*X4] using sorting and cumsum."""
    n = len(support)
    if n == 0:
        return 0.0
    w_sum = np.sum(weights)
    if w_sum <= 1e-12:
        return 0.0
    w = weights / w_sum
    s = support
    
    # Build pairwise sums S = s_i + s_j with probabilities M = w_i * w_j
    S = s[:, None] + s[None, :]
    M = np.outer(w, w)
    
    S_flat = S.flatten()
    M_flat = M.flatten()
    
    # Sort by sum value for efficient prefix sum queries
    sort_idx = np.argsort(S_flat, kind='stable')
    S_sorted = S_flat[sort_idx]
    M_sorted = M_flat[sort_idx]
    M_cumsum = np.cumsum(M_sorted)
    
    # For each (k, l), we need sum of w_i*w_j where s_i+s_j < 2*s_l - s_k
    T = 2.0 * s[None, :] - s[:, None]  # T[k,l] = 2*s_l - s_k
    T_flat = T.flatten()
    
    # Find count of pairs with sum < T[k,l]
    counts = np.searchsorted(S_sorted, T_flat, side='left')
    
    inner_sums = np.zeros_like(T_flat)
    mask = counts > 0
    inner_sums[mask] = M_cumsum[counts[mask] - 1]
    
    W = np.outer(w, w)
    P = np.sum(W * inner_sums.reshape(n, n))
    
    return P


def _pareto_params_to_measure(params, n):
    """
    Convert parametric parameters to (support, weights).
    params: [scale, alpha, log_n_points, ...]
    We use a discrete Pareto distribution:
    - Support points: s_i = scale * (i+1)^(1/alpha) for i=0..n-1 (or similar)
    - Weights: proportional to Pareto density at those points
    Actually, let's use a simpler parametric family:
    - n points: s_i = scale * (i/(n-1))^(1/alpha) for i=0..n-1
    - weights: w_i proportional to i^(-alpha-1) (Pareto tail)
    But we need to be careful about i=0.
    
    Better approach: Use a geometric-like structure with tunable parameters.
    Let's use: s_i = a * r^i for i=0..n-1 (geometric support)
    w_i proportional to b^i (geometric weights)
    Params: [a, log_r, log_b, n]
    """
    a = params[0]  # scale
    log_r = params[1]  # log of ratio
    log_b = params[2]  # log of weight decay
    n_pts = int(np.round(params[3]))
    
    # Clamp n_pts
    n_pts = max(2, min(n_pts, 64))
    
    r = np.exp(log_r)
    b = np.exp(log_b)
    
    # Ensure valid ranges
    r = max(1.01, min(r, 10.0))
    b = max(0.01, min(b, 0.99))
    a = max(0.01, min(a, 100.0))
    
    # Generate geometric support
    idx = np.arange(n_pts, dtype=np.float64)
    support = a * (r ** idx)
    
    # Generate geometric weights
    weights = b ** idx
    
    return support, weights


def _pareto_params_to_measure_v2(params, n_max):
    """
    More flexible parametric family.
    params: [scale, log_ratio, log_weight_decay, n_points, skew]
    Support: s_i = scale * (1 + skew * i)^(1/alpha) 
    Let's use a power-law support with tunable exponent.
    """
    scale = params[0]
    log_ratio = params[1]
    log_wdecay = params[2]
    n_pts = int(np.round(params[3]))
    exponent = params[4] if len(params) > 4 else 1.0
    
    n_pts = max(2, min(n_pts, 48))
    ratio = np.exp(log_ratio)
    wdecay = np.exp(log_wdecay)
    
    ratio = max(1.001, min(ratio, 5.0))
    wdecay = max(0.05, min(wdecay, 0.95))
    scale = max(0.01, min(scale, 50.0))
    exponent = max(0.2, min(exponent, 3.0))
    
    idx = np.arange(n_pts, dtype=np.float64)
    # Power-law support: s_i = scale * (i+1)^exponent
    support = scale * ((idx + 1) ** exponent)
    
    # Exponential-decay weights
    weights = wdecay ** idx
    
    return support, weights


def _objective_pareto(params, n_max):
    """Objective for optimization: negative P."""
    support, weights = _pareto_params_to_measure_v2(params, n_max)
    
    # Validate
    if np.any(support < -1e-9) or np.any(weights < -1e-9):
        return 0.0
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    
    P = _compute_P_fast(support, weights)
    return -P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_P = 0.0
    best_s = None
    best_w = None
    
    n_max = 48
    
    # --- Phase 1: Global search with Differential Evolution over parametric family ---
    # Parameters: [scale, log_ratio, log_wdecay, n_points, exponent]
    bounds = [
        (0.1, 50.0),      # scale
        (0.0, 1.5),       # log_ratio (ratio in [1, 4.5])
        (-3.0, 0.0),      # log_wdecay (wdecay in [0.05, 1])
        (3.0, 48.0),      # n_points
        (0.2, 3.0),       # exponent
    ]
    
    # Initial population: mix of diverse starting points
    rng = np.random.RandomState(42)
    x0_list = []
    
    # Geometric-like starting point
    x0_list.append(np.array([1.0, 0.5, -1.0, 16.0, 1.0]))
    # Heavy tail
    x0_list.append(np.array([5.0, 0.8, -0.5, 24.0, 2.0]))
    # Light tail
    x0_list.append(np.array([0.5, 0.2, -2.0, 8.0, 0.5]))
    # Random
    for _ in range(8):
        x0_list.append(rng.uniform([0.1, 0.0, -3.0, 3.0, 0.2],
                                   [50.0, 1.5, 0.0, 48.0, 3.0]))
    
    x0 = np.array(x0_list)
    
    try:
        res = so.differential_evolution(
            _objective_pareto,
            bounds,
            args=(n_max,),
            seed=42,
            maxiter=150,
            popsize=12,
            tol=1e-5,
            polish=False,
            init=x0,
            updating='deferred'
        )
        
        if res.fun < 0:
            P = -res.fun
            if P > best_P:
                best_P = P
                support, weights = _pareto_params_to_measure_v2(res.x, n_max)
                best_s = support.copy()
                best_w = weights.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 100:
        if best_s is not None:
            return best_s, best_w
        # Fallback
        n = 32
        return np.arange(n, dtype=np.float64), np.ones(n, dtype=np.float64) / n
    
    # --- Phase 2: Local refinement with Nelder-Mead on the parametric space ---
    if best_s is not None:
        # Get the best params from DE
        best_params = res.x.copy()
        
        def _local_obj(p):
            return _objective_pareto(p, n_max)
        
        try:
            res2 = so.minimize(
                _local_obj,
                best_params,
                method='Nelder-Mead',
                args=(n_max,),
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res2.fun < 0:
                P = -res2.fun
                if P > best_P:
                    best_P = P
                    support, weights = _pareto_params_to_measure_v2(res2.x, n_max)
                    best_s = support.copy()
                    best_w = weights.copy()
        except Exception:
            pass
    
    # --- Phase 3: Try a few alternative parametric families for diversity ---
    if time.time() < deadline - 60 and best_s is not None:
        # Try uniform-like with geometric tail
        for n_try in [12, 20, 32]:
            if time.time() > deadline - 30:
                break
            # Create a hybrid: first half uniform, second half geometric decay
            idx = np.arange(n_try, dtype=np.float64)
            support = np.where(idx < n_try//2, idx, n_try//2 + (idx - n_try//2) * 2.0)
            weights = np.where(idx < n_try//2, 1.0, 0.8 ** (idx - n_try//2))
            
            P = _compute_P_fast(support, weights)
            if P > best_P:
                best_P = P
                best_s = support.copy()
                best_w = weights.copy()
    
    # --- Phase 4: Final local polish on the actual (support, weights) if time allows ---
    if time.time() < deadline - 30 and best_s is not None:
        n = len(best_s)
        
        def _obj_direct(params):
            """Optimize directly in (support, weights) space."""
            s = params[:n]
            w = params[n:]
            # Enforce non-negativity
            s = np.maximum(s, 0.0)
            w = np.maximum(w, 1e-12)
            w_sum = np.sum(w)
            if w_sum <= 1e-9:
                return 0.0
            w = w / w_sum
            P = _compute_P_fast(s, w)
            return -P
        
        try:
            initial = np.concatenate([best_s, best_w])
            res3 = so.minimize(
                _obj_direct,
                initial,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9}
            )
            
            if res3.fun < 0:
                P = -res3.fun
                if P > best_P:
                    s_final = np.maximum(res3.x[:n], 0.0)
                    w_final = np.maximum(res3.x[n:], 1e-12)
                    w_sum = np.sum(w_final)
                    if w_sum > 1e-9:
                        best_P = P
                        best_s = s_final
                        best_w = w_final / w_sum
        except Exception:
            pass
    
    # Final validation and return
    if best_s is None:
        n = 32
        return np.arange(n, dtype=np.float64), np.ones(n, dtype=np.float64) / n
    
    # Ensure validity
    best_s = np.maximum(best_s, 0.0)
    best_w = np.maximum(best_w, 0.0)
    w_sum = np.sum(best_w)
    if w_sum <= 1e-9:
        n = 32
        return np.arange(n, dtype=np.float64), np.ones(n, dtype=np.float64) / n
    
    best_w = best_w / w_sum
    
    return best_s, best_w


# EVOLVE-BLOCK-END
