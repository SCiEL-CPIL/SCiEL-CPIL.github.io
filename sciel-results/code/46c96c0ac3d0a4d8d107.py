# EVOLVE-BLOCK-START
import numpy as np
import time

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


def _evaluate_measure_smooth(support, weights, beta=100.0):
    """
    Smooth approximation of P using sigmoid to handle the indicator.
    P_smooth = sum_{k,l,i,j} p_k p_l p_i p_j * sigmoid(beta * (2*s_l - s_k - s_i - s_j))
    """
    n = len(support)
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    p = weights / w_sum
    
    # Compute all 4-tensor terms efficiently
    # threshold_{kl} = 2*s_l - s_k
    # pair_sum_{ij} = s_i + s_j
    # We want sigmoid(beta * (threshold_{kl} - pair_sum_{ij}))
    
    # For memory efficiency with small n, compute in chunks
    # threshold shape: (n, n)
    # pair_sum shape: (n, n)
    
    thresholds = 2.0 * support[None, :] - support[:, None]  # (n, n)
    pair_sums = support[:, None] + support[None, :]  # (n, n)
    
    # diff[k,l,i,j] = thresholds[k,l] - pair_sums[i,j]
    # Shape: (n, n, n, n)
    # For n <= 8, this is at most 4096 elements, fine.
    diff = thresholds[:, :, None, None] - pair_sums[None, None, :, :]
    
    # sigmoid(beta * diff)
    # Clip to avoid overflow
    beta_diff = beta * diff
    beta_diff = np.clip(beta_diff, -50, 50)
    sigmoid_vals = 1.0 / (1.0 + np.exp(-beta_diff))
    
    # Weight tensor: p_k * p_l * p_i * p_j
    weight_tensor = (p[:, None, None, None] * p[None, :, None, None] * 
                     p[None, None, :, None] * p[None, None, None, :])
    
    P_smooth = np.sum(weight_tensor * sigmoid_vals)
    
    return float(P_smooth)


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4].
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
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
    
    # Try structured 3-point and 4-point configurations
    # These are fast to evaluate and may find better local optima
    
    # 3-point: {0, a, 1} with varying a
    for a in np.linspace(0.1, 0.9, 17):
        if time_left() < 30:
            break
        support = np.array([0.0, a, 1.0])
        # Try a few weight configurations
        for w1 in np.linspace(0.1, 0.8, 8):
            for w2 in np.linspace(0.1, 0.8, 8):
                w0 = 1.0 - w1 - w2
                if w0 < 0.01:
                    continue
                weights = np.array([w0, w1, w2])
                score = _evaluate_measure(support, weights)
                update_best(support, weights, score)
    
    # 4-point: {0, a, b, 1} with structured choices
    for a in [0.2, 0.3, 0.4, 0.5]:
        for b in [0.5, 0.6, 0.7, 0.8]:
            if a >= b:
                continue
            if time_left() < 20:
                break
            support = np.array([0.0, a, b, 1.0])
            # Try geometric and uniform weights
            for w1 in [0.15, 0.2, 0.25, 0.3]:
                for w2 in [0.15, 0.2, 0.25, 0.3]:
                    w0 = 0.3
                    w3 = 1.0 - w0 - w1 - w2
                    if w3 < 0.05:
                        continue
                    weights = np.array([w0, w1, w2, w3])
                    score = _evaluate_measure(support, weights)
                    update_best(support, weights, score)
    
    # Use smooth surrogate for optimization with gradient-based method
    if best_support is not None and time_left() > 150:
        n_best = len(best_support)
        
        def smooth_objective(params, beta=50.0):
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
                P = _evaluate_measure_smooth(support, p, beta=beta)
            except Exception:
                return 1.0
            
            return -P
        
        log_w_best = np.log(np.maximum(best_weights, 1e-15))
        x_best = np.concatenate([best_support, log_w_best])
        
        from scipy.optimize import minimize
        # Use L-BFGS-B with the smooth surrogate
        result_smooth = minimize(
            smooth_objective,
            x_best,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        x_smooth = result_smooth.x
        support_smooth = np.maximum(x_smooth[:n_best], 0.0)
        support_smooth = np.sort(support_smooth)
        log_w_smooth = x_smooth[n_best:]
        log_w_shifted_smooth = log_w_smooth - np.max(log_w_smooth)
        w_smooth = np.exp(log_w_shifted_smooth)
        w_sum_smooth = np.sum(w_smooth)
        if w_sum_smooth >= 1e-9:
            p_smooth = w_smooth / w_sum_smooth
            score_smooth = _evaluate_measure(support_smooth, p_smooth)
            update_best(support_smooth, p_smooth, score_smooth)
    
    # Now do a more thorough search using Nelder-Mead on the best configuration
    if best_support is not None and time_left() > 100:
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
            options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
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
    
    # Try adding a 5th point to the best solution if time permits
    if best_support is not None and time_left() > 80 and len(best_support) == 4:
        n_best = len(best_support)
        
        # Add a point in the middle of the largest gap
        gaps = np.diff(best_support)
        max_gap_idx = np.argmax(gaps)
        new_point = best_support[max_gap_idx] + gaps[max_gap_idx] / 2.0
        support_5 = np.insert(best_support, max_gap_idx + 1, new_point)
        support_5 = np.sort(support_5)
        
        # Initialize weights: distribute with small weight for new point
        w_new = np.array([best_weights[0], best_weights[1] * 0.7, 0.1, best_weights[2] * 0.7, best_weights[3]])
        w_new = np.maximum(w_new, 0.01)
        w_new = w_new / np.sum(w_new)
        
        def nm_objective_5(params):
            support = params[:5]
            log_w = params[5:]
            
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
        
        log_w_5 = np.log(np.maximum(w_new, 1e-15))
        x_5 = np.concatenate([support_5, log_w_5])
        
        from scipy.optimize import minimize
        result_5 = minimize(
            nm_objective_5,
            x_5,
            method='Nelder-Mead',
            options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        
        x_final_5 = result_5.x
        support_final_5 = np.maximum(x_final_5[:5], 0.0)
        support_final_5 = np.sort(support_final_5)
        log_w_final_5 = x_final_5[5:]
        log_w_shifted_final_5 = log_w_final_5 - np.max(log_w_final_5)
        w_final_5 = np.exp(log_w_shifted_final_5)
        w_sum_final_5 = np.sum(w_final_5)
        if w_sum_final_5 >= 1e-9:
            p_final_5 = w_final_5 / w_sum_final_5
            score_final_5 = _evaluate_measure(support_final_5, p_final_5)
            update_best(support_final_5, p_final_5, score_final_5)
    
    # Final high-precision polish on the best solution
    if best_support is not None and time_left() > 30:
        n_best = len(best_support)
        
        def polish_objective(params):
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
        result_polish = minimize(
            polish_objective,
            x_best,
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
        )
        
        x_polish = result_polish.x
        support_polish = np.maximum(x_polish[:n_best], 0.0)
        support_polish = np.sort(support_polish)
        log_w_polish = x_polish[n_best:]
        log_w_shifted_polish = log_w_polish - np.max(log_w_polish)
        w_polish = np.exp(log_w_shifted_polish)
        w_sum_polish = np.sum(w_polish)
        if w_sum_polish >= 1e-9:
            p_polish = w_polish / w_sum_polish
            score_polish = _evaluate_measure(support_polish, p_polish)
            update_best(support_polish, p_polish, score_polish)
    
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
