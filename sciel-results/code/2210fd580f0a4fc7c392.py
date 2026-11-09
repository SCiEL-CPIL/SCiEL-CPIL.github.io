# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so

def _compute_P(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] exactly.
    Uses sorted pairwise sums with cumulative weights for O(n^2 log n) per call.
    """
    n = len(support)
    if n == 0:
        return 0.0
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    p = weights / w_sum
    s = support

    # Precompute pairwise sums and product weights
    # sums[i, j] = s[i] + s[j]
    # probs[i, j] = p[i] * p[j]
    sums = s[:, None] + s[None, :]
    w_prod = p[:, None] * p[None, :]

    # Flatten and sort for fast threshold queries
    sums_flat = sums.ravel()
    w_prod_flat = w_prod.ravel()
    
    # Sort sums and accumulate probabilities
    sort_idx = np.argsort(sums_flat)
    sums_sorted = sums_flat[sort_idx]
    w_prod_sorted = w_prod_flat[sort_idx]
    
    # Cumulative sum of probabilities for sorted sums
    # cumsum_w[i] is the total probability of pairs with sum <= sums_sorted[i-1]
    # We use side='left' in searchsorted, so we need cumsum shifted or handled carefully.
    # Let's make cumsum_w of length n*n+1, where cumsum_w[0]=0
    cumsum_w = np.zeros(n * n + 1)
    cumsum_w[1:] = np.cumsum(w_prod_sorted)

    # P = sum_{k, l} p[k] p[l] P(S2 < 2 s[l] - s[k])
    # S2 = X1 + X2
    total = 0.0
    
    # Vectorize over k and l
    # thresholds[k, l] = 2 * s[l] - s[k]
    thresholds = 2.0 * s[None, :] - s[:, None]
    
    # For each threshold, find the index in sorted sums
    # np.searchsorted(sums_sorted, threshold, side='left') gives count of elements < threshold
    idx_arr = np.searchsorted(sums_sorted, thresholds, side='left')
    
    # The probability mass is cumsum_w[idx_arr]
    # Because cumsum_w[i] is sum of w_prod_sorted[0...i-1], which corresponds to sums_sorted[0...i-1]
    # Wait, np.cumsum(w_prod_sorted) gives array C where C[i] = sum(w[0..i]).
    # My cumsum_w construction: cumsum_w[1] = w_prod_sorted[0], cumsum_w[2] = w_prod_sorted[0]+w_prod_sorted[1].
    # So cumsum_w[i] is sum of first i elements.
    # If searchsorted returns idx, it means there are idx elements strictly less than threshold.
    # These are elements at indices 0 to idx-1.
    # Their sum is cumsum_w[idx].
    p_cond = cumsum_w[idx_arr]
    
    total = np.sum(p[:, None] * p[None, :] * p_cond)
    
    return float(total)

def _neg_P_from_params(params, n):
    """
    Objective function for L-BFGS-B.
    params: [deltas_1, ..., deltas_{n-1}, log_weights_1, ..., log_weights_n]
    deltas are differences between support points, ensuring sorted support.
    log_weights ensure positive weights.
    s[0] is fixed to 0.
    """
    # Extract support deltas
    deltas = params[:n-1]
    # Ensure non-negative deltas to keep support sorted and non-decreasing
    deltas = np.maximum(deltas, 0.0)
    
    # Reconstruct support
    s = np.zeros(n)
    if n > 1:
        s[1:] = np.cumsum(deltas)
        
    # Extract log weights
    log_w = params[n-1:]
    # Softmax for stability
    log_w_shifted = log_w - np.max(log_w)
    w = np.exp(log_w_shifted)
    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return 1e10
    p = w / w_sum
    
    # Compute P
    try:
        P = _compute_P(s, p)
    except Exception:
        return 1e10
        
    if not np.isfinite(P):
        return 1e10
        
    return -P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Returns a discrete measure maximizing P[X1+X2+X3 < 2*X4].
    """
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin for 630s limit

    best_P = -1.0
    best_support = np.array([0.0], dtype=np.float64)
    best_weights = np.array([1.0], dtype=np.float64)

    def update_best(sup, wts, P_val):
        nonlocal best_P, best_support, best_weights
        if P_val > best_P:
            best_P = P_val
            best_support = sup.copy()
            best_weights = wts.copy()

    rng = np.random.default_rng(42)

    # Strategy:
    # 1. Optimize for small n (2 to 8) using L-BFGS-B with multiple restarts.
    # 2. Try "growing" the best solution by adding a point and re-optimizing.
    
    # List of n values to try
    n_values = [2, 3, 4, 5, 6, 7, 8]
    
    # We will store the best parameters for each n to potentially grow them
    best_params_for_n = {}
    
    for n in n_values:
        if time.time() > deadline - 20:
            break
        
        # Number of restarts: more for smaller n
        # Budget per restart depends on n. n=2 is very fast, n=8 is slower.
        # _compute_P is O(n^2 log n).
        # Let's estimate time per iteration.
        # For n=8, n^2=64. Very fast.
        # We can afford many restarts.
        n_restarts = max(10, int(100 / n))
        
        best_P_for_this_n = -1.0
        best_params_for_this_n = None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            # Deltas: random positive
            deltas_init = rng.exponential(1.0, size=n-1)
            # Log weights: random normal
            log_w_init = rng.normal(0, 1, size=n)
            
            params_init = np.concatenate([deltas_init, log_w_init])
            
            try:
                res = so.minimize(
                    _neg_P_from_params,
                    params_init,
                    args=(n,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if res.fun < 1e10: # Valid result
                    # Recover support and weights
                    deltas_opt = np.maximum(res.x[:n-1], 0.0)
                    s_opt = np.zeros(n)
                    if n > 1:
                        s_opt[1:] = np.cumsum(deltas_opt)
                    
                    log_w_opt = res.x[n-1:]
                    log_w_shifted = log_w_opt - np.max(log_w_opt)
                    w_opt = np.exp(log_w_shifted)
                    w_opt = w_opt / np.sum(w_opt)
                    
                    P_val = _compute_P(s_opt, w_opt)
                    
                    if P_val > best_P_for_this_n:
                        best_P_for_this_n = P_val
                        best_params_for_this_n = res.x.copy()
                        
                    if P_val > best_P:
                        update_best(s_opt, w_opt, P_val)
                        
            except Exception:
                pass
        
        if best_params_for_this_n is not None:
            best_params_for_n[n] = best_params_for_this_n

    # Growth strategy:
    # Take the best solution for n, add a new point, and optimize n+1.
    # This can help escape local optima.
    for n in sorted(best_params_for_n.keys()):
        if time.time() > deadline - 30:
            break
        
        if n + 1 > 12: # Don't go too big, computation cost increases
            continue
            
        base_params = best_params_for_n[n]
        new_n = n + 1
        
        # Initialize new point in the middle or at the end
        # We need to map old params to new space.
        # Old: [d1..d_{n-1}, lw1..lwn]
        # New: [d1..d_n, lw1..lw_{n+1}]
        
        # Simple strategy: insert a new small delta and split a weight or add a small weight.
        # Let's try adding a point at the end with a small weight.
        # Old deltas: d1...d_{n-1}
        # New deltas: d1...d_{n-1}, d_n (new)
        # Old weights: lw1...lwn
        # New weights: lw1...lwn, lw_{n+1} (new, small)
        
        # To make it a good start, let's place the new support point halfway to the next "virtual" point
        # or just at the max support + small amount.
        
        old_deltas = np.maximum(base_params[:n-1], 0.0)
        old_log_w = base_params[n-1:]
        
        # New delta: small value
        new_delta = rng.uniform(0.01, 0.1)
        # New log weight: small (e.g., -5)
        new_log_w = -5.0
        
        new_params = np.concatenate([old_deltas, [new_delta], old_log_w, [new_log_w]])
        
        # Optimize
        try:
            res = so.minimize(
                _neg_P_from_params,
                new_params,
                args=(new_n,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.fun < 1e10:
                deltas_opt = np.maximum(res.x[:new_n-1], 0.0)
                s_opt = np.zeros(new_n)
                if new_n > 1:
                    s_opt[1:] = np.cumsum(deltas_opt)
                
                log_w_opt = res.x[new_n-1:]
                log_w_shifted = log_w_opt - np.max(log_w_opt)
                w_opt = np.exp(log_w_shifted)
                w_opt = w_opt / np.sum(w_opt)
                
                P_val = _compute_P(s_opt, w_opt)
                
                if P_val > best_P:
                    update_best(s_opt, w_opt, P_val)
                    
        except Exception:
            pass

    # Final validation
    if len(best_support) == 0 or len(best_weights) == 0:
        best_support = np.array([0.0], dtype=np.float64)
        best_weights = np.array([1.0], dtype=np.float64)

    # Ensure valid output format
    best_support = np.asarray(best_support, dtype=np.float64).flatten()
    best_weights = np.asarray(best_weights, dtype=np.float64).flatten()

    # Remove any NaN or inf
    mask = np.isfinite(best_support) & np.isfinite(best_weights)
    best_support = best_support[mask]
    best_weights = best_weights[mask]

    # Ensure non-negative
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)

    # Ensure sum of weights > 0
    if np.sum(best_weights) <= 1e-9:
        best_support = np.array([0.0], dtype=np.float64)
        best_weights = np.array([1.0], dtype=np.float64)

    return best_support, best_weights

# EVOLVE-BLOCK-END
