# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _evaluate_measure(support, weights):
    """
    Compute P = Prob[X1 + X2 + X3 < 2*X4] exactly in float64.
    Returns the probability as a float.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    # Normalize weights
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    p = weights / w_sum
    
    # Build all pairwise sums s_i + s_j with probabilities p_i * p_j
    pair_sums = support[:, None] + support[None, :]
    pair_probs = p[:, None] * p[None, :]
    
    pair_sums_flat = pair_sums.ravel()
    pair_probs_flat = pair_probs.ravel()
    
    # Sort by pair_sums
    sort_idx = np.argsort(pair_sums_flat)
    sorted_sums = pair_sums_flat[sort_idx]
    sorted_probs = pair_probs_flat[sort_idx]
    
    # Cumulative sum of probs
    cum_probs = np.cumsum(sorted_probs)
    
    # thresholds shape (n, n): thresholds[k, l] = 2*s_l - s_k
    thresholds = 2.0 * support[None, :] - support[:, None]
    T_flat = thresholds.ravel()
    
    # Find index where sorted_sums < T_flat (strict inequality)
    idx = np.searchsorted(sorted_sums, T_flat, side='left')
    
    cdf_vals = np.zeros(len(idx), dtype=np.float64)
    mask = idx > 0
    cdf_vals[mask] = cum_probs[idx[mask] - 1]
    
    cdf_vals = cdf_vals.reshape(n, n)
    
    # p_k p_l
    prob_kl = p[:, None] * p[None, :]
    
    P = np.sum(prob_kl * cdf_vals)
    
    return float(P)

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
            best_support = sup.copy()
            best_weights = wts.copy()
    
    def time_left():
        return deadline - time.time()
    
    # Strategy 1: Quick two-point scan to get a good starting point
    # For support [0, 1] with weight p on 1:
    # P = p*(1-p)^2*(1+2p)
    # Maximize: f(p) = p*(1-p)^2*(1+2p)
    # Optimum at p = (1 + sqrt(17)) / 8.0
    p_opt_two = (1 + np.sqrt(17)) / 8.0
    support_two = np.array([0.0, 1.0])
    weights_two = np.array([1.0 - p_opt_two, p_opt_two])
    score_two = _evaluate_measure(support_two, weights_two)
    update_best(support_two, weights_two, score_two)
    
    # Also scan a range around the optimum
    for p_val in np.linspace(0.4, 0.8, 81):
        if time_left() < 60:
            break
        support = np.array([0.0, 1.0])
        weights = np.array([1.0 - p_val, p_val])
        score = _evaluate_measure(support, weights)
        update_best(support, weights, score)
    
    # Strategy 2: Stochastic Coordinate Descent for n=3,4,5,6
    # This avoids the pitfalls of Nelder-Mead on non-smooth landscapes.
    # We treat support and log-weights as parameters.
    
    np.random.seed(42)
    
    # Define the optimization function
    def optimize_local(support_init, log_w_init, n, max_iters=2000):
        support = support_init.copy()
        log_w = log_w_init.copy()
        
        # Initial evaluation
        w = np.exp(log_w - np.max(log_w))
        w_sum = np.sum(w)
        if w_sum < 1e-9:
            return support, log_w, -1.0
        p = w / w_sum
        current_score = _evaluate_measure(support, p)
        
        step_size_sup = 0.5
        step_size_w = 0.5
        
        for it in range(max_iters):
            if time_left() < 30:
                break
            
            # Randomly select a coordinate to update
            # 50% chance support, 50% chance weight
            if np.random.rand() < 0.5:
                # Update support
                idx = np.random.randint(n)
                old_val = support[idx]
                # Geometric step to allow large jumps
                # Support must be >= 0
                # Try moving up or down
                direction = np.random.choice([-1, 1])
                new_val = old_val * (1.0 + direction * step_size_sup)
                
                # Ensure non-negative
                if new_val < 0:
                    new_val = old_val * (1.0 - step_size_sup * 0.5)
                    if new_val < 0:
                        continue
                
                support[idx] = new_val
                # Re-sort support and corresponding weights? 
                # Actually, the measure is defined by pairs (s_i, w_i). 
                # Sorting is not strictly necessary for the calculation, 
                # but it helps if we assume an ordering. 
                # However, the calculation is symmetric in indices except for the sum.
                # Let's just evaluate.
                
                w = np.exp(log_w - np.max(log_w))
                w_sum = np.sum(w)
                if w_sum < 1e-9:
                    support[idx] = old_val
                    continue
                p = w / w_sum
                new_score = _evaluate_measure(support, p)
                
                if new_score > current_score:
                    current_score = new_score
                    # Reduce step size slightly if we improved? Or keep it?
                    # Let's keep it constant for now, maybe reduce if stuck.
                else:
                    support[idx] = old_val
            else:
                # Update weight (log weight)
                idx = np.random.randint(n)
                old_val = log_w[idx]
                direction = np.random.choice([-1, 1])
                new_val = old_val + direction * step_size_w
                
                log_w[idx] = new_val
                w = np.exp(log_w - np.max(log_w))
                w_sum = np.sum(w)
                if w_sum < 1e-9:
                    log_w[idx] = old_val
                    continue
                p = w / w_sum
                new_score = _evaluate_measure(support, p)
                
                if new_score > current_score:
                    current_score = new_score
                else:
                    log_w[idx] = old_val
            
            # Adaptive step size: if no improvement in last 50 iters, reduce step size
            if it % 50 == 49:
                # Simple heuristic: if score hasn't changed much, reduce step
                # We don't track history here, so just a slow decay
                step_size_sup *= 0.95
                step_size_w *= 0.95
                
        return support, log_w, current_score

    for n in [3, 4, 5, 6]:
        if time_left() < 120:
            break
        
        # Number of restarts
        n_restarts = max(3, int(10 * 4.0 / n))
        
        for restart in range(n_restarts):
            if time_left() < 60:
                break
            
            # Generate diverse initializations
            if restart == 0:
                # Start near the best two-point solution extended
                if best_support is not None and len(best_support) == 2:
                    support_init = np.array([0.0, 1.0] + list(np.linspace(0.5, 2.0, n-2)))
                    support_init = np.sort(support_init)
                else:
                    support_init = np.sort(np.array([0.0, 1.0] + list(np.linspace(0.2, 1.8, n-2))))
                log_w_init = np.zeros(n)
            elif restart == 1:
                # Concentrated near 0
                support_init = np.sort(np.array([0.0, 0.1, 0.5] + list(np.random.uniform(0, 1, max(0, n-3)))))
                log_w_init = np.random.randn(n)
            elif restart == 2:
                # Spread out
                support_init = np.sort(np.random.uniform(0, 3, n))
                support_init[0] = 0.0  # Ensure at least one point at 0
                log_w_init = np.random.randn(n)
            elif restart == 3:
                # Bimodal
                half = n // 2
                support_init = np.sort(np.concatenate([
                    np.random.uniform(0, 0.5, half),
                    np.random.uniform(1.5, 2.5, n - half)
                ]))
                support_init[0] = 0.0
                log_w_init = np.random.randn(n)
            else:
                # Random
                support_init = np.sort(np.random.uniform(0, 2, n))
                support_init[0] = 0.0
                log_w_init = np.random.randn(n)
            
            # Ensure support is valid
            support_init = np.maximum(support_init, 0.0)
            support_init = np.sort(support_init)
            
            # Run local optimization
            sup_opt, log_w_opt, score_opt = optimize_local(support_init, log_w_init, n, max_iters=1000)
            
            if score_opt > -0.5: # Valid score
                w_opt = np.exp(log_w_opt - np.max(log_w_opt))
                w_sum = np.sum(w_opt)
                if w_sum > 1e-9:
                    p_opt = w_opt / w_sum
                    # Ensure support is sorted for consistency (though not strictly needed for eval)
                    # But we should return a clean array
                    order = np.argsort(sup_opt)
                    sup_final = sup_opt[order]
                    p_final = p_opt[order]
                    update_best(sup_final, p_final, score_opt)
    
    # Strategy 3: Final refinement on the best found
    if best_support is not None and time_left() > 60:
        n_best = len(best_support)
        log_w_best = np.log(np.maximum(best_weights, 1e-15))
        
        # Run a longer local search
        sup_ref, log_w_ref, score_ref = optimize_local(best_support, log_w_best, n_best, max_iters=5000)
        
        if score_ref > best_score:
            w_ref = np.exp(log_w_ref - np.max(log_w_ref))
            w_sum = np.sum(w_ref)
            if w_sum > 1e-9:
                p_ref = w_ref / w_sum
                order = np.argsort(sup_ref)
                sup_final = sup_ref[order]
                p_final = p_ref[order]
                update_best(sup_final, p_final, score_ref)
    
    # Fallback if nothing found
    if best_support is None:
        support = np.array([0.0, 1.0])
        weights = np.array([1.0 - p_opt_two, p_opt_two])
        best_support = support
        best_weights = weights
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
