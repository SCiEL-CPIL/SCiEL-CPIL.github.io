# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

def _evaluate_measure(support, weights):
    """
    Compute P = Prob[X1 + X2 + X3 < 2*X4] exactly in float64.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    p = weights / w_sum
    
    # Build all pairwise sums
    pair_sums = support[:, None] + support[None, :]
    pair_probs = p[:, None] * p[None, :]
    
    pair_sums_flat = pair_sums.ravel()
    pair_probs_flat = pair_probs.ravel()
    
    sort_idx = np.argsort(pair_sums_flat)
    sorted_sums = pair_sums_flat[sort_idx]
    sorted_probs = pair_probs_flat[sort_idx]
    
    cum_probs = np.cumsum(sorted_probs)
    
    # thresholds[k, l] = 2*s_l - s_k
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
    that maximizes P[X1 + X2 + X3 < 2*X4].
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s limit
    
    best_score = -1.0
    best_support = None
    best_weights = None
    
    def update_best(sup, wts, score):
        nonlocal best_score, best_support, best_weights
        if score > best_score:
            best_score = score
            best_support = np.array(sup, dtype=np.float64)
            best_weights = np.array(wts, dtype=np.float64)
    
    def time_left():
        return deadline - time.time()
    
    # Analytic 2-point optimum: support {0, 1}, weight p* on 1 where p* = (1+sqrt(17))/8
    p_opt_two = (1 + np.sqrt(17)) / 8.0
    support_two = np.array([0.0, 1.0])
    weights_two = np.array([1.0 - p_opt_two, p_opt_two])
    score_two = _evaluate_measure(support_two, weights_two)
    update_best(support_two, weights_two, score_two)
    
    # Objective for differential evolution: minimize -P
    def objective(params, n):
        """
        params: [s_0, s_1, ..., s_{n-1}, w_0, w_1, ..., w_{n-1}]
        """
        support = params[:n]
        weights = params[n:]
        
        # Clamp support to [0, inf)
        support = np.maximum(support, 0.0)
        support = np.sort(support)
        
        # Clamp weights to non-negative, then normalize
        weights = np.maximum(weights, 0.0)
        w_sum = np.sum(weights)
        if w_sum < 1e-9:
            return 1.0
        p = weights / w_sum
        
        try:
            P = _evaluate_measure(support, p)
        except Exception:
            return 1.0
        
        return -P
    
    # Differential evolution for n=3, 4, 5, 6
    for n in [3, 4, 5, 6]:
        if time_left() < 90:
            break
        
        # Bounds for support: [0, 3]
        # Bounds for weights: [0, 1]
        bounds = [(0.0, 3.0)] * n + [(0.0, 1.0)] * n
        
        # Multiple restarts
        n_restarts = 2 if n <= 4 else 1
        
        for restart in range(n_restarts):
            if time_left() < 60:
                break
            
            # Generate initial population hints
            if restart == 0 and best_support is not None and len(best_support) >= 2:
                # Start near the 2-point solution
                if n == 3:
                    init_s = np.array([0.0, 0.5, 1.0])
                elif n == 4:
                    init_s = np.array([0.0, 0.3, 0.7, 1.0])
                elif n == 5:
                    init_s = np.array([0.0, 0.2, 0.4, 0.7, 1.0])
                else:
                    init_s = np.linspace(0.0, 1.0, n)
                init_w = np.ones(n) / n
            else:
                init_s = np.sort(np.random.uniform(0, 2, n))
                init_s[0] = 0.0
                init_w = np.random.dirichlet(np.ones(n))
            
            x0 = np.concatenate([init_s, init_w])
            
            # Run differential evolution
            result = differential_evolution(
                objective,
                bounds,
                args=(n,),
                x0=x0,
                seed=42 + restart * 100,
                maxiter=200,
                popsize=15,
                tol=1e-8,
                polish=True,
                init='sobol' if restart == 0 else 'latinhypercube'
            )
            
            if time_left() < 30:
                break
            
            x_opt = result.x
            support_opt = np.maximum(x_opt[:n], 0.0)
            support_opt = np.sort(support_opt)
            w_opt = np.maximum(x_opt[n:], 0.0)
            w_sum = np.sum(w_opt)
            if w_sum < 1e-9:
                continue
            p_opt = w_opt / w_sum
            
            score = _evaluate_measure(support_opt, p_opt)
            update_best(support_opt, p_opt, score)
    
    # Final local polish with Nelder-Mead on the best solution
    if best_support is not None and time_left() > 60:
        n_best = len(best_support)
        
        def nm_objective(params):
            support = params[:n_best]
            log_w = params[n_best:]
            
            support = np.maximum(support, 0.0)
            support = np.sort(support)
            
            log_w_shifted = log_w - np.max(log_w)
            w = np.exp(log_w_shifted)
            w_sum = np.sum(w)
            if w_sum < 1e-9:
                return 1.0
            p = w / w_sum
            
            try:
                P = _evaluate_measure(support, p)
            except Exception:
                return 1.0
            
            return -P
        
        log_w_best = np.log(np.maximum(best_weights, 1e-15))
        x_best = np.concatenate([best_support, log_w_best])
        
        from scipy.optimize import minimize
        result_final = minimize(
            nm_objective,
            x_best,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        
        x_final = result_final.x
        support_final = np.maximum(x_final[:n_best], 0.0)
        support_final = np.sort(support_final)
        log_w_final = x_final[n_best:]
        log_w_shifted_final = log_w_final - np.max(log_w_final)
        w_final = np.exp(log_w_shifted_final)
        w_sum_final = np.sum(w_final)
        if w_sum_final >= 1e-9:
            p_final = w_final / w_sum_final
            score_final = _evaluate_measure(support_final, p_final)
            update_best(support_final, p_final, score_final)
    
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

# EVOLVE-BLOCK-END
