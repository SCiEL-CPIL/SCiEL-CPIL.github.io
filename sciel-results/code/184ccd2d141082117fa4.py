# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _compute_P(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] for i.i.d. X1..X4 from the discrete measure
    defined by support and weights (weights need not be normalized).
    
    Uses the exact formula:
    P = sum_{k,l} w_k w_l * (sum_{i,j: s_i + s_j < 2*s_l - s_k} w_i w_j)
    where the inner sums are over the normalized weights.
    
    For efficiency, we precompute the pairwise sums and use a sorted approach.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    # Normalize weights
    w = weights / weights.sum()
    
    # Compute all pairwise sums s_i + s_j and their probabilities w_i * w_j
    # We need for each (k, l) pair: sum over (i,j) where s_i + s_j < 2*s_l - s_k of w_i * w_j
    
    # Strategy: 
    # 1. Compute all n^2 pairwise sums and sort them
    # 2. For each (k, l), we need the cumulative probability of pairwise sums < threshold
    
    # Build pairwise sums
    pair_sums = support[:, None] + support[None, :]  # shape (n, n)
    pair_probs = w[:, None] * w[None, :]  # shape (n, n)
    
    # Flatten and sort
    flat_sums = pair_sums.flatten()
    flat_probs = pair_probs.flatten()
    
    # Sort by sum
    sort_idx = np.argsort(flat_sums, kind='stable')
    sorted_sums = flat_sums[sort_idx]
    sorted_probs = flat_probs[sort_idx]
    
    # Cumulative sum of probabilities
    cum_probs = np.cumsum(sorted_probs)
    
    # Now for each (k, l), threshold = 2*s_l - s_k
    # We need sum of probabilities where pair_sum < threshold
    # Use searchsorted to find the index, then take cum_probs[index-1]
    
    # Build all thresholds: for k in range(n), l in range(n): 2*support[l] - support[k]
    # This is an (n, n) array where [k, l] = 2*support[l] - support[k]
    thresholds = 2.0 * support[None, :] - support[:, None]  # shape (n, n), [k, l]
    
    # For each threshold, find how many sorted sums are strictly less than it
    # searchsorted with side='left' gives the first index where sorted_sums >= threshold
    # So the count of elements < threshold is that index
    # We want cumulative probability up to (but not including) threshold
    
    # Use searchsorted: for each threshold, find insertion point
    # We need to do this for n^2 thresholds
    # To speed up, we can process in batches or use broadcasting
    
    # Actually, let's compute this more efficiently
    # For each unique threshold value, we can use searchsorted once
    # But thresholds may not be unique
    
    # Let's just do it with a loop over l (outer) and use vectorized searchsorted
    # Or better: use the fact that we can do searchsorted on the full array
    
    # Method: for each (k, l), threshold = 2*s_l - s_k
    # count = np.searchsorted(sorted_sums, threshold, side='left')
    # prob = cum_probs[count-1] if count > 0 else 0
    
    # Vectorized approach:
    # Create a 2D array of thresholds, then for each row k, we have n thresholds
    # We can use searchsorted which works on 1D arrays
    
    # Let's process row by row (over k)
    P = 0.0
    for k in range(n):
        # thresholds for this k: 2*support - support[k], shape (n,)
        thresh_row = 2.0 * support - support[k]  # shape (n,)
        
        # For each threshold in thresh_row, find count of sorted_sums < threshold
        counts = np.searchsorted(sorted_sums, thresh_row, side='left')
        
        # Get cumulative probabilities: where count > 0, take cum_probs[count-1], else 0
        # Handle count=0 case
        inner_probs = np.zeros(n)
        mask = counts > 0
        if np.any(mask):
            inner_probs[mask] = cum_probs[counts[mask] - 1]
        
        # P contribution from this k: sum over l of w[k]*w[l]*inner_probs[l]
        P += w[k] * np.dot(w, inner_probs)
    
    return P


def _compute_P_fast(support, weights):
    """
    Faster computation using a different approach.
    """
    n = len(support)
    if n == 0:
        return 0.0
    
    w = weights / weights.sum()
    
    # Alternative: iterate over l (X4 value), and for each l, compute
    # sum over k of w_k * P[X1+X2 < 2*s_l - s_k]
    # where P[X1+X2 < t] can be precomputed for all relevant t values
    
    # The relevant t values are 2*s_l - s_k for all k, l
    # There are at most n^2 distinct values, but we can sort and use cumulative
    
    # Precompute pairwise sums and their cumulative distribution
    pair_sums = support[:, None] + support[None, :]
    pair_probs = w[:, None] * w[None, :]
    
    flat_sums = pair_sums.flatten()
    flat_probs = pair_probs.flatten()
    
    sort_idx = np.argsort(flat_sums, kind='stable')
    sorted_sums = flat_sums[sort_idx]
    sorted_probs = flat_probs[sort_idx]
    cum_probs = np.cumsum(sorted_probs)
    
    P = 0.0
    for l in range(n):
        # For this l, threshold for each k is 2*s_l - s_k
        thresh = 2.0 * support[l] - support  # shape (n,)
        counts = np.searchsorted(sorted_sums, thresh, side='left')
        inner_probs = np.zeros(n)
        mask = counts > 0
        if np.any(mask):
            inner_probs[mask] = cum_probs[counts[mask] - 1]
        P += w[l] * np.dot(w, inner_probs)
    
    return P


def _optimize_weights(support, initial_weights, max_iters=200, time_limit=550):
    """
    Optimize weights given fixed support using coordinate descent / gradient-free approach.
    Uses a simple iterative method: for each weight, try to adjust it to improve P.
    """
    n = len(support)
    w = initial_weights.copy()
    w = np.maximum(w, 1e-15)
    w = w / w.sum()
    
    start_time = time.time()
    best_w = w.copy()
    best_P = _compute_P_fast(support, w)
    
    # Simple coordinate ascent: for each coordinate, try small perturbations
    step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001]
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            
            current_P = _compute_P_fast(support, w)
            
            for step in step_sizes:
                # Try increasing w[i] (decrease others proportionally)
                w_trial = w.copy()
                w_trial[i] += step
                # Renormalize
                w_trial = w_trial / w_trial.sum()
                P_trial = _compute_P_fast(support, w_trial)
                
                if P_trial > current_P + 1e-12:
                    w = w_trial
                    current_P = P_trial
                    improved = True
                    break
                
                # Try decreasing w[i]
                if w[i] > 1e-10:
                    w_trial = w.copy()
                    w_trial[i] -= step
                    if w_trial[i] < 0:
                        w_trial[i] = 1e-15
                    w_trial = w_trial / w_trial.sum()
                    P_trial = _compute_P_fast(support, w_trial)
                    
                    if P_trial > current_P + 1e-12:
                        w = w_trial
                        current_P = P_trial
                        improved = True
                        break
        
        if not improved:
            # Try random restarts within the simplex
            for _ in range(10):
                if time.time() - start_time > time_limit:
                    break
                w_rand = np.random.dirichlet(np.ones(n))
                P_rand = _compute_P_fast(support, w_rand)
                if P_rand > current_P + 1e-12:
                    w = w_rand
                    current_P = P_rand
                    improved = True
                    break
        
        if not improved:
            break
        
        if current_P > best_P:
            best_P = current_P
            best_w = w.copy()
    
    return best_w, best_P


def _optimize_support(support, weights, max_iters=50, time_limit=550):
    """
    Optimize support points given fixed weights.
    """
    n = len(support)
    s = support.copy()
    w = weights / weights.sum()
    
    start_time = time.time()
    best_s = s.copy()
    best_P = _compute_P_fast(s, w)
    
    step_sizes = [1.0, 0.5, 0.1, 0.05, 0.01]
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            
            current_P = _compute_P_fast(s, w)
            
            for step in step_sizes:
                # Try increasing s[i]
                s_trial = s.copy()
                s_trial[i] += step
                if s_trial[i] < 0:
                    continue
                P_trial = _compute_P_fast(s_trial, w)
                
                if P_trial > current_P + 1e-12:
                    s = s_trial
                    current_P = P_trial
                    improved = True
                    break
                
                # Try decreasing s[i]
                if s[i] > step:
                    s_trial = s.copy()
                    s_trial[i] -= step
                    if s_trial[i] < 0:
                        s_trial[i] = 0
                    P_trial = _compute_P_fast(s_trial, w)
                    
                    if P_trial > current_P + 1e-12:
                        s = s_trial
                        current_P = P_trial
                        improved = True
                        break
        
        if not improved:
            break
        
        if current_P > best_P:
            best_P = current_P
            best_s = s.copy()
    
    return best_s, best_P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    global_start = time.time()
    total_time_limit = 620  # Leave 10s margin
    
    # Start with a heavy-tailed design
    # Use a mixture: some mass on small values, some on larger values
    # Design: support = [0, a, b, c, d, e, f, g, h, i] with geometric or log-spaced values
    
    best_support = None
    best_weights = None
    best_P = 0.0
    
    # Initial candidate designs
    candidates = []
    
    # Candidate 1: Uniform on {0, 1, ..., 31} (baseline)
    n1 = 32
    support1 = np.arange(n1, dtype=np.float64)
    weights1 = np.ones(n1, dtype=np.float64) / n1
    candidates.append((support1, weights1))
    
    # Candidate 2: Heavy-tailed - geometric spacing
    n2 = 20
    support2 = np.array([0.0] + [1.5**i for i in range(1, n2)])
    # Weights: decay but with more mass on larger values
    weights2 = np.array([0.3] + [0.7 * 0.85**(i-1) for i in range(1, n2)])
    weights2 = weights2 / weights2.sum()
    candidates.append((support2, weights2))
    
    # Candidate 3: Bimodal - mass at 0 and at large value
    n3 = 10
    support3 = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
    weights3 = np.array([0.25, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05])
    weights3 = weights3 / weights3.sum()
    candidates.append((support3, weights3))
    
    # Candidate 4: Log-spaced
    n4 = 25
    support4 = np.logspace(0, 2, n4)  # 1 to 100
    support4[0] = 0.0
    weights4 = np.ones(n4) / n4
    candidates.append((support4, weights4))
    
    # Candidate 5: More aggressive heavy tail
    n5 = 15
    support5 = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 
                         512.0, 1024.0, 2048.0, 4096.0, 8192.0])
    weights5 = np.array([0.15, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05,
                         0.05, 0.02, 0.02, 0.01, 0.00])
    weights5 = np.maximum(weights5, 1e-10)
    weights5 = weights5 / weights5.sum()
    candidates.append((support5, weights5))
    
    # Evaluate all candidates
    for support, weights in candidates:
        if time.time() - global_start > total_time_limit - 10:
            break
        P = _compute_P_fast(support, weights)
        if P > best_P:
            best_P = P
            best_support = support.copy()
            best_weights = weights.copy()
    
    # Now optimize the best candidate
    if best_support is not None and time.time() - global_start < total_time_limit - 30:
        # Optimize weights
        opt_weights, opt_P = _optimize_weights(
            best_support, best_weights, 
            max_iters=100, 
            time_limit=min(200, total_time_limit - (time.time() - global_start) - 20)
        )
        if opt_P > best_P:
            best_P = opt_P
            best_weights = opt_weights
        
        # Optimize support
        if time.time() - global_start < total_time_limit - 20:
            opt_support, opt_P2 = _optimize_support(
                best_support, best_weights,
                max_iters=30,
                time_limit=min(100, total_time_limit - (time.time() - global_start) - 10)
            )
            if opt_P2 > best_P:
                best_P = opt_P2
                best_support = opt_support
        
        # Final joint optimization: alternate weight and support optimization
        if time.time() - global_start < total_time_limit - 10:
            for _ in range(3):
                if time.time() - global_start > total_time_limit - 10:
                    break
                # Optimize weights
                w_new, P_w = _optimize_weights(
                    best_support, best_weights,
                    max_iters=30,
                    time_limit=min(50, total_time_limit - (time.time() - global_start) - 10)
                )
                if P_w > best_P:
                    best_P = P_w
                    best_weights = w_new
                
                # Optimize support
                if time.time() - global_start > total_time_limit - 10:
                    break
                s_new, P_s = _optimize_support(
                    best_support, best_weights,
                    max_iters=20,
                    time_limit=min(30, total_time_limit - (time.time() - global_start) - 5)
                )
                if P_s > best_P:
                    best_P = P_s
                    best_support = s_new
    
    # Validate and return
    if best_support is None or best_weights is None:
        # Fallback to baseline
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    
    # Ensure validity
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / best_weights.sum()
    
    # Final check
    final_P = _compute_P_fast(best_support, best_weights)
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
