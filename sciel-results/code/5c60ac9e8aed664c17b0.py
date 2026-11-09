# EVOLVE-BLOCK-START
import numpy as np
import time

def _evaluate_measure(support, weights):
    """
    Compute P = Prob[X1 + X2 + X3 < 2*X4] exactly in float64.
    Returns the probability as a float.
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


def _optimize_weights(support, w0, max_iter=2000, tol=1e-12):
    """
    Optimize weights on a fixed support using projected gradient ascent on the simplex.
    Returns optimized weights (normalized to sum to 1) and the score.
    """
    n = len(support)
    w = w0.copy()
    w = np.maximum(w, 1e-15)
    w = w / w.sum()
    
    best_w = w.copy()
    best_score = _evaluate_measure(support, w)
    
    # Precompute pair sums sorted (these don't change with weights)
    pair_sums = support[:, None] + support[None, :]
    pair_sums_flat = pair_sums.ravel()
    sort_idx = np.argsort(pair_sums_flat)
    sorted_sums = pair_sums_flat[sort_idx]
    
    thresholds = 2.0 * support[None, :] - support[:, None]
    T_flat = thresholds.ravel()
    idx = np.searchsorted(sorted_sums, T_flat, side='left')
    
    # For each (k,l), the CDF contribution is sum of p_i*p_j where s_i+s_j < T[k,l]
    # We can precompute the matrix M[k,l,i,j] = 1 if s_i+s_j < 2*s_l - s_k
    # But that's n^4. Instead, use the sorted approach.
    
    # Gradient computation:
    # P = sum_{k,l} p_k p_l * F(T_{kl})
    # where F(T) = sum_{i,j: s_i+s_j < T} p_i p_j
    # dP/dp_m = sum_{k,l} [delta_{km} p_l + p_k delta_{lm}] * F(T_{kl})
    #          + sum_{k,l} p_k p_l * dF/dp_m
    # dF(T_{kl})/dp_m = sum_j [delta_{mj} + delta_{jm}] * 1_{s_m+s_j < T_{kl}} 
    #                 = 2 * sum_j 1_{s_m+s_j < T_{kl}} - 1_{2*s_m < T_{kl}}
    # Wait, F(T) = sum_{i,j} p_i p_j * 1_{s_i+s_j < T}
    # dF/dp_m = sum_j p_j * 1_{s_m+s_j < T} + sum_i p_i * 1_{s_i+s_m < T}
    #          = 2 * sum_j p_j * 1_{s_m+s_j < T}  (since the sums are symmetric)
    #          But wait, when i=m and j=m, it's counted twice? No:
    # dF/dp_m = sum_j p_j * 1_{s_m+s_j < T} + sum_i p_i * 1_{s_i+s_m < T}
    # These are the same sum, so = 2 * sum_j p_j * 1_{s_m+s_j < T}
    
    # Precompute for each m and each (k,l) pair: G[m, k, l] = sum_j p_j * 1_{s_m+s_j < 2*s_l - s_k}
    # This is n * n * n = n^3 which is fine for n <= 10.
    
    # Actually, let's compute gradient numerically for simplicity and robustness,
    # but use a smarter approach: finite differences on the simplex.
    
    # Better: use the analytical gradient.
    # Let's precompute indicator matrices.
    # For each m, for each (k,l): ind[m,k,l] = sum_j 1_{s_m+s_j < 2*s_l - s_k}
    # But we need weighted sum: sum_j p_j * 1_{s_m+s_j < 2*s_l - s_k}
    
    # Let's build a 3D array: for each m, a (n,n) matrix of G[m,k,l]
    # G[m] is computed as: for each j, if s_m + s_j < 2*s_l - s_k for various (k,l)
    
    # Simpler: just do numerical gradient with central differences on log-space?
    # No, let's do it properly.
    
    # Precompute: for each m, the (n,n) matrix H[m,k,l] = sum_j p_j * 1_{s_m+s_j < 2*s_l-s_k}
    # We can compute this by: for each m, for each j, the condition s_m+s_j < 2*s_l-s_k
    # is equivalent to 2*s_l > s_k + s_m + s_j
    # For fixed m,j, this is a condition on (k,l).
    
    # Let's just compute the gradient by evaluating P at w and w+eps*e_m (projected).
    # For n <= 10, this is 10 extra evaluations per gradient, each O(n^2 log n^2).
    # With 2000 iterations, that's 20000 evaluations. Each is fast for n=10.
    
    eps = 1e-6
    for it in range(max_iter):
        # Compute gradient via finite differences
        grad = np.zeros(n)
        for m in range(n):
            w_plus = w.copy()
            w_plus[m] += eps
            w_plus = np.maximum(w_plus, 0)
            w_plus = w_plus / w_plus.sum()
            
            w_minus = w.copy()
            w_minus[m] -= eps
            w_minus = np.maximum(w_minus, 0)
            w_minus = w_minus / w_minus.sum()
            
            f_plus = _evaluate_measure(support, w_plus)
            f_minus = _evaluate_measure(support, w_minus)
            grad[m] = (f_plus - f_minus) / (2 * eps)
        
        # Project gradient to tangent space of simplex
        # Tangent: sum of components = 0
        grad_tangent = grad - np.mean(grad)
        
        # Adaptive step size
        if it < 10:
            lr = 0.1
        else:
            lr = 0.05
        
        w_new = w + lr * grad_tangent
        w_new = np.maximum(w_new, 0)
        w_sum = w_new.sum()
        if w_sum < 1e-9:
            break
        w_new = w_new / w_sum
        
        # Check for improvement
        score_new = _evaluate_measure(support, w_new)
        if score_new > best_score:
            best_score = score_new
            best_w = w_new.copy()
        
        if abs(score_new - best_score) < tol:
            break
        
        w = w_new
    
    return best_w, best_score


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
            best_support = sup.copy()
            best_weights = wts.copy()
    
    def time_left():
        return deadline - time.time()
    
    # Initial good starting point: 2-point solution
    p_opt_two = (1 + np.sqrt(17)) / 8.0
    support_init = np.array([0.0, 1.0])
    weights_init = np.array([1.0 - p_opt_two, p_opt_two])
    score_init = _evaluate_measure(support_init, weights_init)
    update_best(support_init, weights_init, score_init)
    
    # Try a few structured 3-point starts
    for s_mid in [0.3, 0.5, 0.7, 1.0]:
        if time_left() < 60:
            break
        for p0 in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for p2 in [0.1, 0.2, 0.3, 0.4, 0.5]:
                if p0 + p2 >= 0.95:
                    continue
                p1 = 1.0 - p0 - p2
                sup = np.array([0.0, s_mid, 1.0])
                wts = np.array([p0, p1, p2])
                sc = _evaluate_measure(sup, wts)
                update_best(sup, wts, sc)
    
    # Main search: iterate over different support configurations
    # Use a "grow and shrink" approach
    
    np.random.seed(123)
    
    # List of support configurations to try
    configs = []
    
    # 3-point configs
    for s1 in [0.2, 0.4, 0.6, 0.8, 1.2, 1.5]:
        for s2 in [1.0, 1.5, 2.0, 2.5]:
            if s1 < s2:
                configs.append(np.array([0.0, s1, s2]))
    
    # 4-point configs
    for s1 in [0.2, 0.4, 0.6]:
        for s2 in [0.8, 1.0, 1.2]:
            for s3 in [1.5, 2.0, 2.5]:
                if s1 < s2 < s3:
                    configs.append(np.array([0.0, s1, s2, s3]))
    
    # 5-point configs
    for s1 in [0.2, 0.4]:
        for s2 in [0.6, 0.8]:
            for s3 in [1.0, 1.2]:
                for s4 in [1.5, 2.0]:
                    if s1 < s2 < s3 < s4:
                        configs.append(np.array([0.0, s1, s2, s3, s4]))
    
    # 6-point configs
    for s1 in [0.2, 0.3]:
        for s2 in [0.5, 0.7]:
            for s3 in [0.9, 1.1]:
                for s4 in [1.3, 1.5]:
                    for s5 in [1.8, 2.2]:
                        if s1 < s2 < s3 < s4 < s5:
                            configs.append(np.array([0.0, s1, s2, s3, s4, s5]))
    
    # Also add the current best as a config
    if best_support is not None:
        configs.append(best_support.copy())
    
    # Shuffle configs for diversity
    np.random.shuffle(configs)
    
    # For each config, optimize weights
    for sup in configs:
        if time_left() < 30:
            break
        
        n = len(sup)
        
        # Multiple weight initializations
        init_weights = [
            np.ones(n) / n,
            np.array([0.5] + [0.5/(n-1)] * (n-1)),
            np.array([0.3, 0.3] + [0.4/(n-2)] * (n-2)) if n > 2 else np.ones(n)/n,
            np.random.dirichlet(np.ones(n)),
            np.random.dirichlet(np.ones(n) * 0.5),
            np.array([0.1, 0.1] + [0.8/(n-2)] * (n-2)) if n > 2 else np.ones(n)/n,
        ]
        
        best_w_for_sup = None
        best_sc_for_sup = -1.0
        
        for w0 in init_weights:
            if time_left() < 15:
                break
            
            w_opt, sc = _optimize_weights(sup, w0, max_iter=1500, tol=1e-12)
            if sc > best_sc_for_sup:
                best_sc_for_sup = sc
                best_w_for_sup = w_opt
        
        if best_w_for_sup is not None:
            update_best(sup, best_w_for_sup, best_sc_for_sup)
            
            # Try to grow: add a new support point
            if n < 8 and time_left() > 60:
                # Add points at various locations
                for new_s in [0.1, 0.3, 0.5, 0.7, 1.3, 1.7, 2.5, 3.0]:
                    if time_left() < 30:
                        break
                    
                    # Insert new point, maintaining sorted order
                    sup_new = np.sort(np.append(sup, new_s))
                    
                    # Initialize weights: give small weight to new point
                    w_new = np.zeros(len(sup_new))
                    # Map old weights to new array
                    for i, old_s in enumerate(sup):
                        # Find index in new sup
                        idx = np.searchsorted(sup_new, old_s)
                        w_new[idx] = best_w_for_sup[i]
                    # Give small weight to new point
                    # Find the new point's index
                    new_idx = np.searchsorted(sup_new, new_s)
                    w_new[new_idx] = 0.05
                    w_new = np.maximum(w_new, 1e-6)
                    w_new = w_new / w_new.sum()
                    
                    w_opt2, sc2 = _optimize_weights(sup_new, w_new, max_iter=1000, tol=1e-12)
                    if sc2 > best_score:
                        update_best(sup_new, w_opt2, sc2)
            
            # Try to shrink: remove a support point with smallest weight
            if n > 2 and time_left() > 60:
                min_idx = np.argmin(best_w_for_sup)
                if best_w_for_sup[min_idx] < 0.1:  # Only if weight is small
                    sup_shrink = np.delete(sup, min_idx)
                    w_shrink = np.delete(best_w_for_sup, min_idx)
                    w_shrink = w_shrink / w_shrink.sum()
                    
                    w_opt3, sc3 = _optimize_weights(sup_shrink, w_shrink, max_iter=1000, tol=1e-12)
                    if sc3 > best_score:
                        update_best(sup_shrink, w_opt3, sc3)
    
    # Final refinement: take the best solution and do one more round of weight optimization
    if best_support is not None and time_left() > 30:
        w_final, sc_final = _optimize_weights(best_support, best_weights, max_iter=3000, tol=1e-14)
        update_best(best_support, w_final, sc_final)
    
    # Fallback if nothing found
    if best_support is None:
        support = np.array([0.0, 1.0])
        weights = np.array([1.0 - p_opt_two, p_opt_two])
        best_support = support
        best_weights = weights
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
