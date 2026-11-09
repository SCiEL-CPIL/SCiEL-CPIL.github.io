import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _evaluate_measure(support, weights):
    """
    Compute P = Prob[X1 + X2 + X3 < 2*X4] exactly in float64.
    Uses sorted pairwise sums + cumulative probabilities + searchsorted for O(n^2 log n) per call.
    For small n, uses a direct O(n^4) loop which is faster due to low overhead.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    p = weights / w_sum
    
    # For very small n, direct computation is often faster in Python due to NumPy overhead
    if n <= 8:
        # Direct: for each (k,l), compute CDF(2*s_l - s_k) by summing over (i,j)
        P = 0.0
        # Precompute pair sums and probs for speed? No, direct is fine for n<=8 (4096 ops max)
        # But let's stick to the vectorized one for consistency unless it's really slow.
        # Actually, for n=3, n^4=81. Vectorized involves sorting 9 elements.
        # Let's just use the vectorized version, it's robust.
        pass

    # Sort support and weights to ensure consistency for searchsorted
    if not np.all(support[:-1] <= support[1:]):
        order = np.argsort(support)
        support = support[order]
        p = p[order]
    
    pair_sums = support[:, None] + support[None, :]
    pair_probs = p[:, None] * p[None, :]
    
    pair_sums_flat = pair_sums.ravel()
    pair_probs_flat = pair_probs.ravel()
    
    sort_idx = np.argsort(pair_sums_flat, kind='stable')
    sorted_sums = pair_sums_flat[sort_idx]
    sorted_probs = pair_probs_flat[sort_idx]
    
    cum_probs = np.cumsum(sorted_probs)
    
    # thresholds[k, l] = 2*s_l - s_k
    # Note: in the formula P = sum_{k,l} w_k w_l P[X1+X2 < 2s_l - s_k]
    # So for fixed k (X3=s_k) and l (X4=s_l), we need P[X1+X2 < 2s_l - s_k]
    thresholds = 2.0 * support[None, :] - support[:, None]
    T_flat = thresholds.ravel()
    
    idx = np.searchsorted(sorted_sums, T_flat, side='left')
    
    cdf_vals = np.zeros(len(idx), dtype=np.float64)
    mask = idx > 0
    cdf_vals[mask] = cum_probs[idx[mask] - 1]
    
    cdf_vals = cdf_vals.reshape(n, n)
    
    prob_kl = p[:, None] * p[None, :]
    
    P = np.sum(prob_kl * cdf_vals)
    
    return float(P)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin before 600s kill

    def time_left():
        return deadline - time.time()

    best_score = -1.0
    best_support = None
    best_weights = None
    
    def update_best(sup, wts, score):
        nonlocal best_score, best_support, best_weights
        if score > best_score:
            best_score = score
            best_support = np.array(sup, dtype=np.float64)
            best_weights = np.array(wts, dtype=np.float64)
    
    # 1. Analytic 2-point optimum
    # Maximize p(1-p)^2(1+2p) for support {0,1}
    # Derivative leads to 8p^3 - 9p^2 + 1 = 0 => (p-1)(8p^2 - p - 1) = 0
    # p = (1 + sqrt(33))/16
    p_opt_two = (1.0 + np.sqrt(33.0)) / 16.0
    support_two = np.array([0.0, 1.0])
    weights_two = np.array([1.0 - p_opt_two, p_opt_two])
    score_two = _evaluate_measure(support_two, weights_two)
    update_best(support_two, weights_two, score_two)
    
    # 2. Differential Evolution for n=3, 4, 5, 6
    # Strategy: Allow support to range up to 3.0 to capture the "large K" effect.
    # The insight is that a large support point K with small weight p contributes ~p to the probability
    # (when X4=K, 2K is huge, so X1+X2 < 2K is almost always true), while the cost is small.
    
    for n in [3, 4, 5, 6]:
        if time_left() < 60:
            break
        
        # Parameters: n support points, n-1 weights (last weight is 1 - sum(others))
        # To handle constraints simply in DE, we optimize raw weights and normalize,
        # but we must ensure positivity. 
        # Better: optimize log-weights? Or just bound weights to [eps, 1] and normalize.
        # Let's stick to the approach: optimize s_0..s_{n-1} and w_0..w_{n-2}. w_{n-1} = 1 - sum.
        # This is tricky with DE because of the sum constraint.
        # Alternative: Optimize n supports and n raw weights, then normalize weights.
        # This removes the simplex constraint from the optimizer's burden, only positivity remains.
        
        def objective(params):
            s = params[:n]
            w_raw = params[n:]
            
            # Support constraints: s >= 0
            if np.any(s < -1e-9):
                return 1.0
            
            # Weight constraints: w_raw > 0
            if np.any(w_raw < 1e-9):
                return 1.0
            
            # Normalize weights
            w = w_raw / np.sum(w_raw)
            
            # Ensure support is sorted for the evaluator (though evaluator sorts internally,
            # it's good practice to keep it consistent for the optimizer's view)
            s_sorted, w_sorted = zip(*sorted(zip(s, w)))
            
            try:
                P = _evaluate_measure(np.array(s_sorted), np.array(w_sorted))
            except Exception:
                return 1.0
            
            return -P
        
        # Bounds
        # Support: [0, 3.0] (allowing for the large K effect)
        # Weights: [1e-4, 1.0] (raw weights, will be normalized)
        bounds = [(0.0, 3.0)] * n + [(1e-4, 1.0)] * n
        
        n_restarts = 5 if n <= 4 else 3
        
        for restart in range(n_restarts):
            if time_left() < 30:
                break
            
            # Initialization strategies
            if restart == 0:
                # Start near 2-point solution, spread out
                s_init = np.sort(np.linspace(0.0, 1.0, n))
                w_init = np.ones(n) / n
            elif restart == 1:
                # Random in [0, 1]
                s_init = np.sort(np.random.uniform(0, 1, n))
                w_init = np.random.dirichlet(np.ones(n))
            elif restart == 2:
                # One point large (the K strategy)
                s_init = np.sort(np.concatenate([[0.0, 1.0], np.random.uniform(1.5, 3.0, n-2)]))
                w_init = np.random.dirichlet(np.ones(n) * 0.5) # More uniform-ish
            elif restart == 3:
                # Two points large
                s_init = np.sort(np.concatenate([[0.0], np.random.uniform(0.5, 1.5, n-2), np.random.uniform(2.0, 3.0)]))
                w_init = np.random.dirichlet(np.ones(n))
            else:
                # Random wide
                s_init = np.sort(np.random.uniform(0, 3, n))
                w_init = np.random.dirichlet(np.ones(n) * 2)
            
            x0 = np.concatenate([s_init, w_init])
            
            try:
                result = differential_evolution(
                    objective,
                    bounds,
                    x0=x0,
                    seed=42 + restart * 100 + n * 1000,
                    maxiter=200,
                    popsize=20,
                    tol=1e-8,
                    polish=True,
                    init='sobol' if restart == 0 else 'latinhypercube'
                )
            except Exception:
                continue
            
            if time_left() < 10:
                break
            
            x_opt = result.x
            s_opt = x_opt[:n]
            w_opt_raw = x_opt[n:]
            
            # Validate
            if np.any(s_opt < -1e-9) or np.any(w_opt_raw < 1e-9):
                continue
            
            w_opt = w_opt_raw / np.sum(w_opt_raw)
            
            # Sort by support
            order = np.argsort(s_opt)
            s_final = s_opt[order]
            w_final = w_opt[order]
            
            score = _evaluate_measure(s_final, w_final)
            update_best(s_final, w_final, score)
        
        # Local refinement with SLSQP on the best solution of this size
        if best_support is not None and len(best_support) == n and time_left() > 30:
            # SLSQP requires a different parameterization to handle the simplex constraint properly
            # or we use the raw weight approach and add constraints.
            # Let's use the raw weight approach with constraints: w_raw > 0.
            # And support s > 0.
            
            def slsqp_objective(params):
                s = params[:n]
                w_raw = params[n:]
                if np.any(s < 0) or np.any(w_raw < 1e-6):
                    return 1e10
                w = w_raw / np.sum(w_raw)
                order = np.argsort(s)
                s_s = s[order]
                w_s = w[order]
                try:
                    P = _evaluate_measure(s_s, w_s)
                except Exception:
                    return 1e10
                return -P
            
            # Initialize from best
            s_init = best_support.copy()
            w_init = best_weights.copy()
            # SLSQP might struggle with normalization if we don't constrain it.
            # Let's stick to the raw weights.
            x0 = np.concatenate([s_init, w_init])
            
            # Constraints:
            # s_i >= 0
            # w_j >= 0 (handled by bounds mostly, but SLSQP bounds are [0, inf])
            
            cons = []
            # No inequality constraints needed if we use bounds for non-negativity.
            # But we need to ensure sum(w) > 0, which is guaranteed if w >= 0 and not all 0.
            # Bounds: s in [0, 3], w in [1e-6, 10]
            
            bounds_slq = [(0.0, 3.0)] * n + [(1e-6, 10.0)] * n
            
            try:
                result_slq = minimize(
                    slsqp_objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds_slq,
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
            except Exception:
                continue
            
            if time_left() < 5:
                break
            
            x_final = result_slq.x
            s_final = x_final[:n]
            w_final_raw = x_final[n:]
            
            if np.any(s_final < 0) or np.any(w_final_raw < 1e-9):
                continue
            
            w_final = w_final_raw / np.sum(w_final_raw)
            order = np.argsort(s_final)
            s_final = s_final[order]
            w_final = w_final[order]
            
            score = _evaluate_measure(s_final, w_final)
            update_best(s_final, w_final, score)

    # Final Nelder-Mead polish on the overall best solution
    if best_support is not None and time_left() > 30:
        n_best = len(best_support)
        
        def nm_objective(params):
            s = params[:n_best]
            w_raw = params[n_best:]
            if np.any(s < 0) or np.any(w_raw < 1e-9):
                return 1e10
            w = w_raw / np.sum(w_raw)
            order = np.argsort(s)
            s_s = s[order]
            w_s = w[order]
            try:
                P = _evaluate_measure(s_s, w_s)
            except Exception:
                return 1e10
            return -P
        
        s_init = best_support.copy()
        w_init = best_weights.copy()
        x0 = np.concatenate([s_init, w_init])
        
        try:
            result_nm = minimize(
                nm_objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
        except Exception:
            pass
        else:
            x_final = result_nm.x
            s_final = x_final[:n_best]
            w_final_raw = x_final[n_best:]
            
            if np.any(s_final < 0) or np.any(w_final_raw < 1e-9):
                pass
            else:
                w_final = w_final_raw / np.sum(w_final_raw)
                order = np.argsort(s_final)
                s_final = s_final[order]
                w_final = w_final[order]
                
                # Ensure no negative support
                if np.any(s_final < 0):
                    pass
                else:
                    score = _evaluate_measure(s_final, w_final)
                    update_best(s_final, w_final, score)

    # Fallback
    if best_support is None:
        best_support = support_two
        best_weights = weights_two
    
    # Ensure validity: no NaN, no inf, non-negative, positive sum
    best_support = np.nan_to_num(best_support, nan=0.0, posinf=0.0, neginf=0.0)
    best_weights = np.nan_to_num(best_weights, nan=0.0, posinf=0.0, neginf=0.0)
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    w_sum = np.sum(best_weights)
    if w_sum < 1e-9:
        best_support = support_two
        best_weights = weights_two
    
    return best_support, best_weights
