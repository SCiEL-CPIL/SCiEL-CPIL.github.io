# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _evaluate_measure(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] for the given discrete measure.
    support: (n,) array of non-negative floats
    weights: (n,) array of non-negative floats (will be normalized)
    Returns: float P
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    # Normalize weights
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    
    # Compute all pairwise sums S_ij = s_i + s_j
    # S has shape (n, n)
    S = support[:, None] + support[None, :]
    
    # The condition is s_i + s_j < 2*s_l - s_k
    # Let T_kl = 2*s_l - s_k. T has shape (n, n) with T[k, l] = 2*s_l - s_k
    # We need to sum over k, l, i, j such that S[i, j] < T[k, l]
    # P = sum_k sum_l w_k w_l * (sum_i sum_j w_i w_j * 1[S[i, j] < T[k, l]])
    
    # To compute efficiently:
    # For each pair (k, l), we need the cumulative probability of S being less than T[k, l].
    # Since S is fixed, we can sort S and use searchsorted if we had a CDF.
    # However, S is a matrix of values. 
    # Alternative: 
    # Let's iterate over k and l. n is expected to be small (e.g., 10-20).
    # n^2 iterations. Inside each, we need sum_{i,j} w_i w_j [S_ij < T_kl].
    # This is a dot product of w_i w_j (flattened) with the indicator matrix.
    
    # Let's precompute the weighted indicator sums?
    # No, T_kl varies.
    
    # Efficient approach:
    # Flatten S to S_flat of length n^2.
    # Flatten w to w_flat of length n.
    # w_pair = w[:, None] * w[None, :] -> shape (n, n)
    # w_pair_flat = w_pair.flatten()
    
    # For each k, l:
    # target = 2 * support[l] - support[k]
    # mask = S_flat < target
    # prob_kl = np.dot(w_pair_flat, mask)
    
    # Total P = sum_k sum_l w_k w_l * prob_kl
    
    # Complexity: n^2 (outer) * n^2 (inner dot/mask) = n^4 operations.
    # If n=10, 10^4 = 10,000 ops. Fast.
    # If n=20, 20^4 = 160,000 ops. Fast.
    # If n=50, 50^4 = 6.25 million ops. Still okay in numpy if vectorized properly?
    # Actually, creating the mask is O(n^2). Dot is O(n^2).
    # Looping n^2 times in Python is slow if n is large.
    # Let's vectorize the outer loop if possible.
    
    # T = 2 * support[None, :] - support[:, None]  # Shape (n, n), T[k, l]
    # We need for each element T[k, l], the sum of w_pair_flat where S_flat < T[k, l].
    
    # This is a bit tricky to fully vectorize without sorting or broadcasting huge arrays.
    # Given the budget (630s) and that Nelder-Mead might call this ~100-500 times,
    # and n will be kept small (e.g., 10-15), the Python loop is acceptable.
    
    S_flat = S.flatten()
    w_pair = w[:, None] * w[None, :]
    w_pair_flat = w_pair.flatten()
    
    T = 2.0 * support[None, :] - support[:, None]
    
    P = 0.0
    # Vectorize the inner sum over i,j for all k,l at once?
    # We can't easily broadcast S_flat (n^2,) against T (n, n) to get a 3D array of indicators (n, n, n^2) without huge memory.
    # But n is small.
    # Let's try to be smart:
    # Sort S_flat and compute weights of sorted S_flat? No, weights are not sorted with S.
    
    # Let's just do the loop. It's simple and robust for small n.
    for k in range(n):
        for l in range(n):
            target = T[k, l]
            # Strict inequality
            mask = S_flat < target
            prob_kl = np.dot(w_pair_flat, mask)
            P += w[k] * w[l] * prob_kl
            
    return P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Optimize the discrete measure using Nelder-Mead.
    """
    start_time = time.time()
    deadline = start_time + 600  # Safety margin
    
    # Initial guess: Uniform on {0, 1, ..., n-1}
    n = 12  # Small n to keep optimization fast and allow more iterations
    
    # Parameters to optimize:
    # support: n values (constrained to be non-negative and sorted? 
    #           Sorting support doesn't change the measure if we sort weights accordingly, 
    #           but Nelder-Mead works better with unconstrained or simple constraints.
    #           We can parameterize support as exp(support_log) to ensure positivity,
    #           but we also need them to be distinct? No, duplicates are allowed in support 
    #           as long as weights handle it, but it's redundant.
    #           Let's just optimize support directly and project to [0, inf) by clipping.
    #           Actually, negative support is invalid.
    #           Let's use a transformation: support = exp(support_raw). 
    #           This ensures support > 0.
    #           But the optimal support might include 0. 
    #           The problem says support >= -1e-9. So 0 is allowed.
    #           Let's try optimizing support directly and clip to 0.
    
    # Weights: optimize log_weights to ensure positivity.
    # log_w = log_weights_raw
    # w = exp(log_w) / sum(exp(log_w))
    
    # Total parameters: n (support) + n (log_weights) = 2n.
    
    x0 = np.zeros(2 * n)
    # Initial support: 0, 1, ..., n-1
    x0[:n] = np.arange(n)
    # Initial log_weights: log(1/n) for all
    x0[n:] = np.log(1.0 / n)
    
    def objective(params):
        if time.time() > deadline:
            return 1e9  # Penalty for timeout
        
        support_raw = params[:n]
        log_w_raw = params[n:]
        
        # Enforce support >= 0
        support = np.maximum(support_raw, 0.0)
        
        # Enforce weights > 0 via softmax-like normalization of exp(log_w)
        # To avoid overflow in exp, subtract max
        log_w_max = np.max(log_w_raw)
        w_unnorm = np.exp(log_w_raw - log_w_max)
        w_sum = np.sum(w_unnorm)
        if w_sum == 0:
            return 1e9
        weights = w_unnorm / w_sum
        
        # Calculate P
        p_val = _evaluate_measure(support, weights)
        
        # We want to maximize P, so minimize -P
        return -p_val

    best_x = x0
    best_val = objective(x0)
    
    # Nelder-Mead options
    options = {
        'maxiter': 2000,
        'xatol': 1e-4,
        'fatol': 1e-6,
        'adaptive': True
    }
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options=options)
        if res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception:
        pass
    
    # Extract best parameters
    support = np.maximum(best_x[:n], 0.0)
    log_w_raw = best_x[n:]
    log_w_max = np.max(log_w_raw)
    w_unnorm = np.exp(log_w_raw - log_w_max)
    weights = w_unnorm / np.sum(w_unnorm)
    
    # Ensure validity
    if np.any(support < -1e-9) or np.any(weights < -1e-9):
        # Fallback to uniform
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        
    return support, weights

# EVOLVE-BLOCK-END
