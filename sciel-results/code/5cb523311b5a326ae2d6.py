# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_P_fast(support, weights):
    """
    Fast computation of P[X1+X2+X3 < 2*X4] using sorting and binary search.
    Complexity: O(n^2 log n)
    """
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    total = w.sum()
    if total <= 1e-9:
        return 0.0
    p = w / total
    n = len(s)
    
    if n == 0:
        return 0.0

    # Pairwise sums and probabilities
    # s[:, None] + s[None, :] gives shape (n, n)
    pair_sums = s[:, None] + s[None, :]
    pair_probs = p[:, None] * p[None, :]
    
    # Flatten for sorting
    flat_sums = pair_sums.flatten()
    flat_probs = pair_probs.flatten()
    
    # Sort sums and get corresponding probabilities
    sort_idx = np.argsort(flat_sums)
    sorted_sums = flat_sums[sort_idx]
    sorted_probs = flat_probs[sort_idx]
    
    # Cumulative sum of probabilities
    cum_probs = np.cumsum(sorted_probs)
    
    P = 0.0
    # Iterate over k and l
    # Threshold for (k, l) is 2*s[l] - s[k]
    # We need sum of pair_probs where pair_sums < threshold
    
    # Vectorize the searchsorted part if possible, but loop is n^2 which is small enough
    # n=20 -> 400 iterations. n=100 -> 10000 iterations.
    # searchsorted is O(log N) where N = n^2.
    
    # Precompute thresholds
    # thresholds[k, l] = 2*s[l] - s[k]
    thresholds = 2.0 * s[None, :] - s[:, None]
    
    # For each threshold, find the number of pair_sums strictly less than it
    # np.searchsorted(sorted_sums, thresholds, side='left') gives index of first element >= threshold
    # So the number of elements < threshold is the index itself.
    
    # searchsorted works on 1D arrays. We can pass a 2D array of thresholds?
    # Yes, np.searchsorted(sorted_sums, thresholds, side='left') will return an array of indices same shape as thresholds.
    
    indices = np.searchsorted(sorted_sums, thresholds, side='left')
    
    # Now we need the sum of sorted_probs up to indices.
    # If index is 0, sum is 0.
    # If index is len, sum is cum_probs[-1].
    # We can use np.take with a safe index, or handle boundaries.
    
    # Create a safe index array for cum_probs
    # indices can range from 0 to len(sorted_sums)
    # cum_probs has length len(sorted_sums)
    # If index == len, we want cum_probs[-1]
    # If index < len, we want cum_probs[index-1]
    
    # Let's create an array of sums
    sum_probs = np.zeros_like(indices, dtype=np.float64)
    
    # Mask for indices > 0
    mask_pos = indices > 0
    # For indices > 0, the sum is cum_probs[index-1]
    # We need to be careful if index-1 >= len(cum_probs)
    # But searchsorted returns indices in [0, len(sorted_sums)]
    # So index-1 is in [-1, len-1]
    
    safe_indices = np.where(mask_pos, indices - 1, 0)
    # Clip to valid range [0, len-1]
    safe_indices = np.clip(safe_indices, 0, len(cum_probs)-1)
    
    sum_probs[mask_pos] = cum_probs[safe_indices[mask_pos]]
    # sum_probs[~mask_pos] remains 0.0
    
    # Now P = sum_{k,l} p[k] p[l] sum_probs[k,l]
    P = np.sum(p[:, None] * p[None, :] * sum_probs)
    
    return P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 600.0 - 30.0  # 30s safety margin
    
    best_P = 0.0
    best_support = np.array([0.0, 1.0])
    best_weights = np.array([0.5, 0.5])
    
    # Candidate support structures
    # 1. Geometric progressions
    # 2. Arithmetic progressions
    # 3. Mixed: small cluster + large points
    
    supports_to_try = []
    
    # Geometric
    for n in [3, 4, 5, 6, 8, 10]:
        for base in [1.5, 2.0, 3.0, 5.0]:
            s = np.array([base**i for i in range(n)])
            # Scale to start at 0? Or just positive?
            # Support must be >= 0.
            # Let's try shifting to start at 0 if base > 1
            # s = s - s[0]
            supports_to_try.append(s)
            
    # Arithmetic
    for n in [3, 4, 5, 6, 8, 10, 15]:
        for step in [1.0, 2.0, 0.5]:
            s = np.arange(n) * step
            supports_to_try.append(s)
            
    # Mixed: 0, 1, ..., k, then large jumps
    for n in [5, 6, 8]:
        for k in range(2, n):
            s = np.zeros(n)
            s[:k] = np.arange(k)
            # Add large values
            for j in range(k, n):
                s[j] = 10.0 * (j - k + 1)
            supports_to_try.append(s)
            
    # Optimize weights for each support
    for s in supports_to_try:
        if time.time() > deadline - 60.0:
            break
            
        n = len(s)
        
        # Initial guess: uniform
        w0 = np.ones(n) / n
        
        # Objective function: minimize -P
        def objective(w_flat):
            w = np.abs(w_flat)  # Ensure non-negative
            s_norm = s.copy()
            # Ensure support is non-negative
            if np.any(s_norm < -1e-9):
                return 1.0
            P = _compute_P_fast(s_norm, w)
            return -P
        
        # Constraints: sum(w) = 1 is handled by normalization in _compute_P,
        # but for optimization stability, it's better to optimize on the simplex.
        # L-BFGS-B doesn't handle equality constraints well directly without transformation.
        # We can optimize log-weights or just use the fact that _compute_P normalizes.
        # However, scaling invariance makes the landscape flat in the radial direction.
        # It's better to fix sum(w)=1.
        # Let's use a transformation: w_i = exp(z_i) / sum(exp(z_j)) (Softmax)
        # This ensures w_i > 0 and sum(w)=1.
        
        def objective_softmax(z):
            # Softmax
            exp_z = np.exp(z - np.max(z))
            w = exp_z / exp_z.sum()
            P = _compute_P_fast(s, w)
            return -P
        
        # Initial z: log(w0)
        z0 = np.log(w0)
        
        try:
            res = minimize(objective_softmax, z0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.success or res.fun < -best_P:
                z_opt = res.x
                exp_z = np.exp(z_opt - np.max(z_opt))
                w_opt = exp_z / exp_z.sum()
                
                P_opt = -res.fun
                if P_opt > best_P:
                    best_P = P_opt
                    best_support = s.copy()
                    best_weights = w_opt.copy()
                    
        except Exception:
            pass
            
    # Final check: try a few random supports with optimization
    if time.time() < deadline - 120.0:
        np.random.seed(42)
        for _ in range(20):
            n = np.random.randint(3, 12)
            # Random sorted support
            s_raw = np.sort(np.random.uniform(0, 20, n))
            # Ensure distinct
            s = s_raw
            z0 = np.zeros(n)
            
            def objective_softmax(z):
                exp_z = np.exp(z - np.max(z))
                w = exp_z / exp_z.sum()
                P = _compute_P_fast(s, w)
                return -P
                
            try:
                res = minimize(objective_softmax, z0, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-6})
                if res.fun < -best_P:
                    z_opt = res.x
                    exp_z = np.exp(z_opt - np.max(z_opt))
                    w_opt = exp_z / exp_z.sum()
                    P_opt = -res.fun
                    if P_opt > best_P:
                        best_P = P_opt
                        best_support = s.copy()
                        best_weights = w_opt.copy()
            except Exception:
                pass

    # Ensure validity
    if len(best_support) != len(best_weights):
        best_support = np.array([0.0, 1.0])
        best_weights = np.array([0.5, 0.5])
        
    if np.any(best_support < -1e-9):
        best_support = np.abs(best_support)
        
    if np.any(best_weights < -1e-9):
        best_weights = np.abs(best_weights)
        
    total_w = best_weights.sum()
    if total_w <= 1e-9:
        best_weights = np.ones(len(best_support)) / len(best_support)
    else:
        best_weights = best_weights / total_w
        
    return best_support, best_weights

# EVOLVE-BLOCK-END
