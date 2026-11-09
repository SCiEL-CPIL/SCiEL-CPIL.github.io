# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

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
    
    # Vectorized evaluation for all n
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


def _coordinate_descent(support, weights, max_iter=100, tol=1e-10):
    """
    Coordinate descent on (support, weights) with normalization.
    """
    n = len(support)
    best_P = _evaluate_measure(support, weights)
    best_s = support.copy()
    best_w = weights.copy()
    
    for it in range(max_iter):
        improved = False
        
        # Optimize each support coordinate (keep it sorted)
        for i in range(n):
            old_s = best_s[i]
            lo = 0.0 if i == 0 else best_s[i-1]
            hi = best_s[i+1] if i < n-1 else old_s * 2 + 1.0
            
            candidates = []
            if i == 0:
                candidates.extend([0.0, 0.1, 0.5, old_s])
                hi = max(hi, 3.0)
            else:
                candidates.extend([lo, hi, (lo+hi)/2.0, old_s])
                for f in [0.25, 0.5, 0.75]:
                    candidates.append(lo + f*(hi-lo))
            
            best_trial_P = best_P
            best_trial_s = old_s
            for c in candidates:
                if c < lo - 1e-12 or c > hi + 1e-12:
                    continue
                s_trial = best_s.copy()
                s_trial[i] = c
                s_trial = np.sort(s_trial)
                P_trial = _evaluate_measure(s_trial, best_w)
                if P_trial > best_trial_P:
                    best_trial_P = P_trial
                    best_trial_s = c
            
            if best_trial_s != old_s and best_trial_P > best_P + tol:
                best_s[i] = best_trial_s
                best_s = np.sort(best_s)
                best_P = best_trial_P
                improved = True
        
        # Optimize each weight coordinate (with normalization)
        for i in range(n):
            old_w = best_w[i]
            candidates = [0.0, 1.0, old_w, 0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8]
            best_trial_P = best_P
            best_trial_w = old_w
            for w_trial in candidates:
                if w_trial < 0 or w_trial > 1:
                    continue
                w_trial_arr = best_w.copy()
                others_sum = 1.0 - w_trial
                if others_sum < 1e-12:
                    w_trial_arr[:] = 0.0
                    w_trial_arr[i] = 1.0
                else:
                    other_mask = np.ones(n, dtype=bool)
                    other_mask[i] = False
                    orig_others_sum = np.sum(best_w[other_mask])
                    if orig_others_sum < 1e-15:
                        w_trial_arr[i] = w_trial
                        w_trial_arr[other_mask] = others_sum / (n-1) if n > 1 else 0.0
                    else:
                        w_trial_arr[i] = w_trial
                        w_trial_arr[other_mask] = best_w[other_mask] * (others_sum / orig_others_sum)
                
                P_trial = _evaluate_measure(best_s, w_trial_arr)
                if P_trial > best_trial_P:
                    best_trial_P = P_trial
                    best_trial_w = w_trial
            
            if best_trial_P > best_P + tol:
                w_trial_arr = best_w.copy()
                others_sum = 1.0 - best_trial_w
                if others_sum < 1e-12:
                    w_trial_arr[:] = 0.0
                    w_trial_arr[i] = 1.0
                else:
                    other_mask = np.ones(n, dtype=bool)
                    other_mask[i] = False
                    orig_others_sum = np.sum(best_w[other_mask])
                    if orig_others_sum < 1e-15:
                        w_trial_arr[i] = best_trial_w
                        w_trial_arr[other_mask] = others_sum / (n-1) if n > 1 else 0.0
                    else:
                        w_trial_arr[i] = best_trial_w
                        w_trial_arr[other_mask] = best_w[other_mask] * (others_sum / orig_others_sum)
                best_w = w_trial_arr
                best_P = best_trial_P
                improved = True
        
        if not improved:
            break
    
    return best_s, best_w, best_P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4].
    """
    start_time = time.time()
    deadline = start_time + 600
    
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
    
    # Analytic 2-point optimum: support {0, 1}, weight p* on 1
    p_opt_two = (1.0 + np.sqrt(33.0)) / 16.0
    support_two = np.array([0.0, 1.0])
    weights_two = np.array([1.0 - p_opt_two, p_opt_two])
    score_two = _evaluate_measure(support_two, weights_two)
    update_best(support_two, weights_two, score_two)
    
    # Try various 2-point configurations
    for p_try in [0.3, 0.35, 0.4, 0.42, 0.4215, 0.45, 0.5, 0.55, 0.6, 0.65]:
        for a_try in [0.5, 1.0, 1.5, 2.0]:
            w_try = np.array([1.0 - p_try, p_try])
            s_try = np.array([0.0, a_try])
            sc_try = _evaluate_measure(s_try, w_try)
            update_best(s_try, w_try, sc_try)
    
    # Global optimization using Differential Evolution for n=3 to n=6
    for n in range(3, 7):
        if time_left() < 60:
            break
        
        def objective(vars):
            if time.time() > deadline - 10:
                return 1e10
            
            s = np.zeros(n)
            s[0:n-1] = vars[0:n-1]
            s[n-1] = 1.0
            
            # Sort support and corresponding weights
            idx = np.argsort(s)
            s_sorted = s[idx]
            w_logits = vars[n-1:2*n-1]
            w_logits_sorted = w_logits[idx]
            
            # Softmax for weights
            w_max = np.max(w_logits_sorted)
            w = np.exp(w_logits_sorted - w_max)
            w_sum = np.sum(w)
            if w_sum == 0:
                return 1e10
            w = w / w_sum
            
            # Ensure s >= 0
            s_sorted = np.maximum(s_sorted, 0.0)
            
            # Check validity
            if np.any(np.isnan(s_sorted)) or np.any(np.isnan(w)):
                return 1e10
            
            P = _evaluate_measure(s_sorted, w)
            return -P
        
        # Bounds for DE
        # s_i in [0, 1] for i < n-1
        # w_logits in [-10, 10]
        bounds = [(0, 1)] * (n-1) + [(-10, 10)] * n
        
        maxiter = 800
        popsize = 20
        
        try:
            res = differential_evolution(
                objective, 
                bounds, 
                maxiter=maxiter, 
                popsize=popsize,
                tol=1e-6,
                seed=42,
                init='sobol',
                update_interval=1
            )
            
            if res.fun < 1e9:
                P_val = -res.fun
                if P_val > best_score:
                    best_score = P_val
                    
                    s = np.zeros(n)
                    s[0:n-1] = res.x[0:n-1]
                    s[n-1] = 1.0
                    w_logits = res.x[n-1:2*n-1]
                    
                    idx = np.argsort(s)
                    s_sorted = s[idx]
                    w_logits_sorted = w_logits[idx]
                    
                    w_max = np.max(w_logits_sorted)
                    w = np.exp(w_logits_sorted - w_max)
                    w = w / np.sum(w)
                    s_sorted = np.maximum(s_sorted, 0.0)
                    
                    best_support = s_sorted
                    best_weights = w
        except Exception:
            pass
    
    # Local refinement using Nelder-Mead on the best solution found
    if best_support is not None and time_left() > 30:
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
                return 0.0
            p = w / w_sum
            
            try:
                P = _evaluate_measure(support, p)
            except Exception:
                return 0.0
            
            return -P
        
        log_w_best = np.log(np.maximum(best_weights, 1e-15))
        x_best = np.concatenate([best_support, log_w_best])
        
        # Multiple polish passes
        for polish_pass in range(4):
            if time_left() < 15:
                break
            
            maxiter = 8000 if polish_pass == 0 else 12000
            xatol = 1e-12 if polish_pass == 0 else (1e-14 if polish_pass == 1 else 1e-15)
            fatol = 1e-14 if polish_pass == 0 else (1e-16 if polish_pass == 1 else 1e-17)
            
            try:
                result_final = minimize(
                    nm_objective,
                    x_best,
                    method='Nelder-Mead',
                    options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
                )
            except Exception:
                result_final = None
            
            if result_final is not None:
                x_best = result_final.x
                support_final = np.maximum(x_best[:n_best], 0.0)
                support_final = np.sort(support_final)
                log_w_final = x_best[n_best:]
                log_w_shifted_final = log_w_final - np.max(log_w_final)
                w_final = np.exp(log_w_shifted_final)
                w_sum_final = np.sum(w_final)
                if w_sum_final >= 1e-9:
                    p_final = w_final / w_sum_final
                    score_final = _evaluate_measure(support_final, p_final)
                    update_best(support_final, p_final, score_final)
        
        # Coordinate descent as final refinement
        if time_left() > 20:
            s_cd, w_cd, P_cd = _coordinate_descent(best_support, best_weights, max_iter=100, tol=1e-12)
            update_best(s_cd, w_cd, P_cd)
    
    # Fallback
    if best_support is None:
        best_support = support_two
        best_weights = weights_two
    
    # Ensure validity
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
