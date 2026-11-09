import numpy as np
import time
import random
from scipy.optimize import minimize, differential_evolution

def _calc_prob_fast(support, weights):
    """
    Compute P[X1+X2+X3 < 2*X4] for a discrete measure.
    Vectorized for speed.
    Returns 0 if invalid.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    # Basic validity check for speed
    if np.any(np.isnan(support)) or np.any(np.isnan(weights)):
        return 0.0
    if np.any(np.isinf(support)) or np.any(np.isinf(weights)):
        return 0.0
    if np.any(support < -1e-9) or np.any(weights < -1e-9):
        return 0.0
        
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
        
    p = weights / w_sum
    
    # Pairwise sums matrix: S[i,j] = s_i + s_j
    S = support[:, None] + support[None, :]
    
    # Weights outer product: W[i,j] = p_i * p_j
    W = p[:, None] * p[None, :]
    
    S_flat = S.flatten()
    W_flat = W.flatten()
    
    # Pre-sort S_flat and corresponding W_flat for fast prefix sums
    sort_idx = np.argsort(S_flat)
    S_sorted = S_flat[sort_idx]
    W_sorted = W_flat[sort_idx]
    W_cumsum = np.cumsum(W_sorted)
    W_cumsum_padded = np.concatenate(([0.0], W_cumsum))
    
    # Vectorize over k: for each k, T_k = 2*s_k - s_l for all l
    T_matrix = 2.0 * support[:, None] - support[None, :]
    T_flat = T_matrix.flatten()
    
    # Find indices in S_sorted where S_sorted < T_flat
    indices = np.searchsorted(S_sorted, T_flat, side='left')
    
    # Sum of weights for each (k,l) pair
    sums_for_kl = W_cumsum_padded[indices]
    sums_for_kl = sums_for_kl.reshape(n, n)
    
    # prob3_k = sum_l p_l * sums_for_kl[k,l]
    prob3 = sums_for_kl @ p  # shape (n,)
    
    total_prob = np.dot(p, prob3)
    
    return float(total_prob)

def _validate(support, weights):
    if len(support) != len(weights):
        return False
    if len(support) < 1:
        return False
    if np.any(np.isnan(support)) or np.any(np.isnan(weights)):
        return False
    if np.any(np.isinf(support)) or np.any(np.isinf(weights)):
        return False
    if np.any(support < -1e-9):
        return False
    if np.any(weights < -1e-9):
        return False
    if np.sum(weights) <= 1e-9:
        return False
    return True

def _params_to_measure(params, n):
    """
    Convert unconstrained parameters to valid support and weights.
    params[0:n] -> log(support)
    params[n:2n] -> raw weights (softmax or abs+normalize)
    """
    log_s = params[:n]
    raw_w = params[n:]
    
    # Support: ensure non-negative. log_s can be any real number.
    # exp(log_s) is always positive.
    support = np.exp(log_s)
    
    # Weights: ensure positive. 
    # Using exp(raw_w) ensures positivity.
    exp_w = np.exp(raw_w - np.max(raw_w)) # for numerical stability
    weights = exp_w / np.sum(exp_w)
    
    return support, weights

def _objective(params, n):
    support, weights = _params_to_measure(params, n)
    if not _validate(support, weights):
        return -1.0
    prob = _calc_prob_fast(support, weights)
    return -prob # Minimizing negative probability

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game using multi-phase optimization with diverse initialization.
    """
    start_time = time.time()
    deadline = start_time + 585.0  # Safety margin for 600s limit
    
    best_prob = -1.0
    best_support = None
    best_weights = None
    
    # Helper to update best
    def try_update(support, weights, prob):
        nonlocal best_prob, best_support, best_weights
        if prob > best_prob:
            best_prob = prob
            best_support = support.copy()
            best_weights = weights.copy()
    
    # Initialize with multiple diverse starting points
    initializations = [
        ([0.1, 0.5, 1.0, 2.0, 4.0], [0.1, 0.2, 0.3, 0.25, 0.15]),
        ([0.0, 1.0, 3.0], [0.3, 0.4, 0.3]),
        ([0.5, 1.0, 2.0, 4.0, 8.0], [0.15, 0.2, 0.3, 0.2, 0.15]),
        ([0.2, 0.8, 1.5, 3.0], [0.25, 0.3, 0.25, 0.2]),
        ([1.0, 2.0], [0.5, 0.5]),
    ]
    
    for s_init, w_init in initializations:
        s = np.array(s_init, dtype=np.float64)
        w = np.array(w_init, dtype=np.float64)
        if _validate(s, w):
            prob = _calc_prob_fast(s, w)
            try_update(s, w, prob)
    
    if best_support is None:
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([0.5, 0.5], dtype=np.float64)
        best_prob = _calc_prob_fast(best_support, best_weights)
    
    # Phase 1: Simulated annealing with dynamic n (first 35% of time)
    phase1_deadline = start_time + (deadline - start_time) * 0.35
    T_start = 1.0
    T_end = 0.005
    cooling_rate = 0.997
    
    current_support = best_support.copy()
    current_weights = best_weights.copy()
    current_prob = best_prob
    T = T_start
    step_scale = 1.0
    
    while time.time() < phase1_deadline:
        # Choose move type
        move_type = random.random()
        
        if move_type < 0.55:
            # Perturb support and weights
            n = len(current_support)
            log_s = np.log(np.maximum(current_support, 1e-10))
            log_s += np.random.normal(0, 0.25 * step_scale, size=n)
            new_support = np.exp(log_s)
            new_support = np.maximum(new_support, 0.0)
            
            new_weights = current_weights * np.exp(np.random.normal(0, 0.15 * step_scale, size=n))
            new_weights = np.maximum(new_weights, 1e-10)
            new_weights = new_weights / np.sum(new_weights)
            
            new_prob = _calc_prob_fast(new_support, new_weights)
            
        elif move_type < 0.78 and len(current_support) < 12:
            # Add a support point
            n = len(current_support)
            new_support = current_support.copy()
            new_weights = current_weights.copy()
            
            insert_idx = random.randint(0, n)
            if insert_idx == 0:
                new_val = max(current_support[0] * np.exp(np.random.normal(0, 0.4)), 0.0)
            elif insert_idx == n:
                new_val = max(current_support[-1] * np.exp(np.random.normal(0, 0.4)), 0.0)
            else:
                left = current_support[insert_idx - 1]
                right = current_support[insert_idx]
                new_val = np.exp(np.log(max(left, 1e-10)) + np.random.uniform(0, 1) * (np.log(max(right, 1e-10)) - np.log(max(left, 1e-10))))
                new_val = max(new_val, 0.0)
            
            new_support = np.insert(new_support, insert_idx, new_val)
            
            if insert_idx == 0:
                w_new = new_weights[0] * 0.4
                new_weights[0] -= w_new
                new_weights = np.insert(new_weights, 0, w_new)
            elif insert_idx == n:
                w_new = new_weights[-1] * 0.4
                new_weights[-1] -= w_new
                new_weights = np.insert(new_weights, -1, w_new)
            else:
                w_new = (new_weights[insert_idx - 1] + new_weights[insert_idx]) * 0.2
                new_weights[insert_idx - 1] -= w_new
                new_weights[insert_idx] -= w_new
                new_weights = np.insert(new_weights, insert_idx, w_new)
            
            new_weights = np.maximum(new_weights, 1e-10)
            new_weights = new_weights / np.sum(new_weights)
            
            new_prob = _calc_prob_fast(new_support, new_weights)
            
        elif move_type < 0.92 and len(current_support) > 2:
            # Remove a support point
            n = len(current_support)
            remove_idx = random.randint(0, n - 1)
            
            new_support = np.delete(current_support, remove_idx)
            new_weights = np.delete(current_weights, remove_idx)
            
            if len(new_weights) > 0:
                w_removed = current_weights[remove_idx]
                if remove_idx == 0:
                    new_weights[0] += w_removed
                elif remove_idx == n - 1:
                    new_weights[-1] += w_removed
                else:
                    new_weights[remove_idx - 1] += w_removed * 0.5
                    new_weights[remove_idx] += w_removed * 0.5
            
            new_weights = np.maximum(new_weights, 1e-10)
            new_weights = new_weights / np.sum(new_weights)
            
            new_prob = _calc_prob_fast(new_support, new_weights)
        
        else:
            # Large jump
            n = len(current_support)
            new_support = current_support * np.exp(np.random.normal(0, 0.8, size=n))
            new_support = np.maximum(new_support, 0.0)
            new_weights = np.abs(np.random.normal(1.0, 0.4, size=n)) + 0.05
            new_weights = new_weights / np.sum(new_weights)
            
            new_prob = _calc_prob_fast(new_support, new_weights)
        
        # Accept or reject
        delta = new_prob - current_prob
        if delta > 0 or random.random() < np.exp(delta / max(T, 1e-10)):
            current_support = new_support
            current_weights = new_weights
            current_prob = new_prob
            try_update(current_support, current_weights, current_prob)
            step_scale = min(step_scale * 1.005, 2.5)
        else:
            step_scale = max(step_scale * 0.995, 0.2)
        
        T *= cooling_rate
        if T < T_end:
            T = T_end
    
    # Phase 2: Intensive local optimization (next 35% of time)
    phase2_deadline = start_time + (deadline - start_time) * 0.70
    if best_support is not None and time.time() < phase2_deadline:
        n_best = len(best_support)
        
        # Convert to parameter space
        log_s = np.log(np.maximum(best_support, 1e-10))
        log_w = np.log(np.maximum(best_weights, 1e-10))
        params_best = np.concatenate([log_s, log_w])
        
        # Run multiple local optimizations from different starting points
        n_local = 12
        for i in range(n_local):
            if time.time() > phase2_deadline - 15:
                break
            
            if i == 0:
                params_start = params_best.copy()
            elif i == 1:
                params_start = params_best + np.random.normal(0, 0.5, size=2 * n_best)
            elif i == 2:
                params_start = params_best + np.random.normal(0, 1.0, size=2 * n_best)
            else:
                params_start = params_best + np.random.normal(0, 0.3 * (i - 2), size=2 * n_best)
            
            def obj_func(p):
                return _objective(p, n_best)
            
            try:
                res = minimize(
                    obj_func,
                    params_start,
                    method='Nelder-Mead',
                    options={'maxfev': 8000, 'xatol': 1e-12, 'fatol': 1e-12}
                )
                
                final_params = res.x
                support, weights = _params_to_measure(final_params, n_best)
                
                if _validate(support, weights):
                    prob = _calc_prob_fast(support, weights)
                    try_update(support, weights, prob)
                    
            except Exception:
                pass
        
        # Try adjacent n values
        for n_try in [n_best - 1, n_best + 1, n_best + 2]:
            if n_try < 1 or n_try > 15 or time.time() > phase2_deadline - 15:
                continue
            
            # Create initial params for this n
            if n_try < n_best:
                s_sub = best_support[:n_try]
                w_sub = best_weights[:n_try]
                w_sub = w_sub / np.sum(w_sub)
            else:
                s_sub = best_support.copy()
                w_sub = best_weights.copy()
                # Add points
                for _ in range(n_try - n_best):
                    mid_idx = len(s_sub) // 2
                    if mid_idx > 0 and mid_idx < len(s_sub):
                        new_s = (s_sub[mid_idx - 1] + s_sub[mid_idx]) / 2
                    elif mid_idx == 0:
                        new_s = s_sub[0] * 0.5 if s_sub[0] > 0 else 0.1
                    else:
                        new_s = s_sub[-1] * 1.5
                    new_w = w_sub[mid_idx] * 0.3 if mid_idx < len(w_sub) else 0.1
                    if mid_idx < len(w_sub):
                        w_sub[mid_idx] -= new_w
                    s_sub = np.insert(s_sub, mid_idx, max(new_s, 0.0))
                    w_sub = np.insert(w_sub, mid_idx, new_w)
                    w_sub = np.maximum(w_sub, 1e-10)
                    w_sub = w_sub / np.sum(w_sub)
            
            log_s = np.log(np.maximum(s_sub, 1e-10))
            log_w = np.log(np.maximum(w_sub, 1e-10))
            params_try = np.concatenate([log_s, log_w])
            
            def obj_func_try(p):
                return _objective(p, n_try)
            
            try:
                res = minimize(
                    obj_func_try,
                    params_try,
                    method='Nelder-Mead',
                    options={'maxfev': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
                )
                
                final_params = res.x
                support, weights = _params_to_measure(final_params, n_try)
                
                if _validate(support, weights):
                    prob = _calc_prob_fast(support, weights)
                    try_update(support, weights, prob)
                    
            except Exception:
                pass
    
    # Phase 3: Differential evolution for global search (remaining time)
    if time.time() < deadline - 40:
        n_de_candidates = [3, 4, 5, 6, 7, 8]
        if best_support is not None:
            n_best = len(best_support)
            if n_best not in n_de_candidates:
                n_de_candidates.append(n_best)
        
        # Sort by priority: prefer n values close to best
        n_de_candidates.sort(key=lambda x: abs(x - len(best_support)) if best_support is not None else x)
        
        for n_de in n_de_candidates[:5]:  # Limit to 5 DE runs
            if time.time() > deadline - 40:
                break
            
            bounds = []
            for i in range(n_de):
                bounds.append((np.log(1e-3), np.log(50.0)))  # log support
            for i in range(n_de):
                bounds.append((-12.0, 12.0))  # log weight raw
            
            def de_obj(p):
                return _objective(p, n_de)
                
            try:
                # Use multiple seeds for diversity
                for seed in [42, 123, 456]:
                    if time.time() > deadline - 40:
                        break
                    
                    res_de = differential_evolution(
                        de_obj, 
                        bounds, 
                        maxiter=80, 
                        popsize=20,
                        seed=seed,
                        tol=1e-9,
                        polish=True,
                        init='sobol'
                    )
                    final_params = res_de.x
                    support, weights = _params_to_measure(final_params, n_de)
                    
                    if _validate(support, weights):
                        prob = _calc_prob_fast(support, weights)
                        try_update(support, weights, prob)
                        
            except Exception:
                pass
    
    # Final high-precision polish
    if best_support is not None and time.time() < deadline - 10:
        n_best = len(best_support)
        log_s = np.log(np.maximum(best_support, 1e-10))
        log_w = np.log(np.maximum(best_weights, 1e-10))
        params_best = np.concatenate([log_s, log_w])
        
        def obj_func(p):
            return _objective(p, n_best)
        
        try:
            # Multiple polish attempts
            for attempt in range(3):
                if time.time() > deadline - 10:
                    break
                
                if attempt == 0:
                    p_start = params_best
                else:
                    p_start = params_best + np.random.normal(0, 0.1, size=2 * n_best)
                
                res = minimize(
                    obj_func,
                    p_start,
                    method='Nelder-Mead',
                    options={'maxfev': 15000, 'xatol': 1e-14, 'fatol': 1e-14}
                )
                
                final_params = res.x
                support, weights = _params_to_measure(final_params, n_best)
                
                if _validate(support, weights):
                    prob = _calc_prob_fast(support, weights)
                    try_update(support, weights, prob)
                    
        except Exception:
            pass
    
    # Fallback
    if best_support is None:
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([0.5, 0.5], dtype=np.float64)
    else:
        # Ensure validity one last time
        if not _validate(best_support, best_weights):
            best_support = np.array([0.0, 1.0], dtype=np.float64)
            best_weights = np.array([0.5, 0.5], dtype=np.float64)
            
        # Normalize weights
        w_sum = np.sum(best_weights)
        if w_sum > 1e-9:
            best_weights = best_weights / w_sum
        else:
            best_weights = np.ones(len(best_support)) / len(best_support)
            
        # Ensure support >= 0
        best_support = np.maximum(best_support, 0.0)

    return best_support.astype(np.float64), best_weights.astype(np.float64)

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print("Support:", s)
    print("Weights:", w)
    print("Probability:", _calc_prob_fast(s, w))
