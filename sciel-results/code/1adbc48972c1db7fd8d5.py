# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly in float64."""
    s = support
    w = weights
    n = len(s)
    w_sum = w.sum()
    if w_sum <= 1e-9:
        return 0.0
    w = w / w_sum
    
    # Sort by support value
    order = np.argsort(s)
    s_sorted = s[order]
    w_sorted = w[order]
    
    # Build all pairwise sums and their probabilities
    pair_sums = s_sorted[:, None] + s_sorted[None, :]
    pair_probs = w_sorted[:, None] * w_sorted[None, :]
    
    # Flatten and sort pair sums
    flat_sums = pair_sums.ravel()
    flat_probs = pair_probs.ravel()
    
    sort_idx = np.argsort(flat_sums, kind='mergesort')
    sorted_sums = flat_sums[sort_idx]
    sorted_probs = flat_probs[sort_idx]
    
    # Cumulative sum of probabilities
    cum_probs = np.cumsum(sorted_probs)
    
    # For each (k, l), we need sum of w_i*w_j where s_i+s_j < 2*s_l - s_k
    # threshold = 2*s_l - s_k
    # Use searchsorted to find how many pairs satisfy the strict inequality
    
    # thresholds[k, l] = 2*s_l - s_k
    thresholds = 2.0 * s_sorted[None, :] - s_sorted[:, None]
    
    # For each threshold, find the number of pairs with sum < threshold
    # searchsorted with side='left' gives the index where threshold would be inserted
    # to maintain sorted order, i.e., number of elements < threshold
    # But we need to be careful with the strict inequality
    
    # For each (k,l), find idx = searchsorted(sorted_sums, threshold, side='left')
    # This gives the count of elements strictly less than threshold
    # But if there are elements equal to threshold, we need to exclude them
    
    # More carefully: we want sum of probs where sum < threshold
    # left_idx = searchsorted(sorted_sums, threshold, side='left')  -> count of elements < threshold... 
    # Actually searchsorted(sorted_sums, threshold, side='left') returns the index where threshold 
    # would be inserted to keep order, which is the number of elements <= threshold? No.
    # side='left': returns the first index where sorted_sums[index] >= threshold
    # So elements [0, left_idx) are all < threshold
    # side='right': returns the first index where sorted_sums[index] > threshold
    # So elements [0, right_idx) are all <= threshold
    
    # We want strictly less, so use side='left'
    left_indices = np.searchsorted(sorted_sums, thresholds.ravel(), side='left')
    left_indices = left_indices.reshape(n, n)
    
    # The cumulative probability up to index i is cum_probs[i-1] for i>0, 0 for i=0
    # We need cum_probs[left_indices - 1] where left_indices > 0, else 0
    inner_probs = np.zeros((n, n), dtype=np.float64)
    mask = left_indices > 0
    inner_probs[mask] = cum_probs[left_indices[mask] - 1]
    
    # Now P = sum over k, l of w_k * w_l * inner_probs[k, l]
    # w_sorted[k] * w_sorted[l] * inner_probs[k, l]
    outer_weights = w_sorted[:, None] * w_sorted[None, :]
    P = np.sum(outer_weights * inner_probs)
    
    return P

def _neg_P(params, support):
    """Negative of P for minimization."""
    weights = np.abs(params)  # Ensure non-negative
    # Add small epsilon to avoid zero weights causing issues
    weights = weights + 1e-12
    return -_compute_P(support, weights)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    # Start with a good initial guess
    # Two-point: {0, M} with optimal p
    # Three-point: {0, eps, M}
    
    best_P = 0.0
    best_support = None
    best_weights = None
    
    # Try two-point distribution first
    M = 1.0
    p_opt = (3.0 - np.sqrt(3.0)) / 6.0
    support2 = np.array([0.0, M], dtype=np.float64)
    weights2 = np.array([1.0 - p_opt, p_opt], dtype=np.float64)
    P2 = _compute_P(support2, weights2)
    if P2 > best_P:
        best_P = P2
        best_support = support2.copy()
        best_weights = weights2.copy()
    
    # Try three-point distribution: {0, a, M}
    # The middle point a should be very small so that it acts like 0 for the inequality
    # but can help by providing more "mass" near 0
    
    # Let's try various values of a
    for a in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.5]:
        support3 = np.array([0.0, a, 1.0], dtype=np.float64)
        # Initial weights: put most mass on 0 and M, small on a
        for p0 in [0.3, 0.4, 0.5, 0.6, 0.7]:
            for pa in [0.01, 0.05, 0.1, 0.2]:
                pM = 1.0 - p0 - pa
                if pM < 0:
                    continue
                weights3 = np.array([p0, pa, pM], dtype=np.float64)
                P3 = _compute_P(support3, weights3)
                if P3 > best_P:
                    best_P = P3
                    best_support = support3.copy()
                    best_weights = weights3.copy()
    
    # Now try to optimize weights for the best support found
    if best_support is not None:
        n = len(best_support)
        # Use the best weights as initial guess
        x0 = best_weights.copy()
        
        # Try multiple restarts
        for restart in range(5):
            if time.time() > deadline - 120:
                break
            
            if restart == 0:
                x_start = x0.copy()
            else:
                # Perturb the initial guess
                x_start = x0 + np.random.normal(0, 0.1, size=n)
                x_start = np.abs(x_start)
                x_start = x_start / x_start.sum()
            
            result = minimize(_neg_P, x_start, args=(best_support,), method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if result.success or result.fun < -best_P:
                w_opt = np.abs(result.x)
                w_opt = w_opt + 1e-12
                P_opt = -result.fun
                if P_opt > best_P:
                    best_P = P_opt
                    best_weights = w_opt.copy()
    
    # Try four-point distribution: {0, a, b, M}
    if time.time() < deadline - 180:
        support4 = np.array([0.0, 1e-4, 0.1, 1.0], dtype=np.float64)
        # Initial weights
        w4_init = np.array([0.5, 0.05, 0.1, 0.35], dtype=np.float64)
        P4 = _compute_P(support4, w4_init)
        if P4 > best_P:
            best_P = P4
            best_support = support4.copy()
            best_weights = w4_init.copy()
        
        # Optimize
        result = minimize(_neg_P, w4_init, args=(support4,), method='Nelder-Mead',
                        options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
        if result.fun < -best_P:
            w_opt = np.abs(result.x)
            w_opt = w_opt + 1e-12
            P_opt = -result.fun
            if P_opt > best_P:
                best_P = P_opt
                best_support = support4.copy()
                best_weights = w_opt.copy()
    
    # Normalize weights
    total = best_weights.sum()
    if total > 1e-9:
        best_weights = best_weights / total
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
