# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_P_fast(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] efficiently using vectorized ops."""
    n = len(support)
    # Normalize weights
    w = weights / weights.sum()
    
    # Pairwise sums: s_i + s_j for all (i,j)
    # We need: for each (k,l), sum of w_i*w_j where s_i+s_j < 2*s_l - s_k
    # P = sum_{k,l} w_k * w_l * (sum_{i,j: s_i+s_j < 2*s_l-s_k} w_i*w_j)
    
    # Build pairwise sum matrix
    pair_sums = support[:, None] + support[None, :]  # (n, n)
    pair_weights = w[:, None] * w[None, :]  # (n, n)
    
    # For each (k, l), threshold is 2*s_l - s_k
    # We need sum of pair_weights where pair_sums < threshold
    # This is a 4D problem: sum over k,l of w_k*w_l * (cumulative sum of pair_weights below threshold)
    
    # Sort pair sums to enable efficient cumulative sum
    flat_sums = pair_sums.ravel()  # (n*n,)
    flat_weights = pair_weights.ravel()  # (n*n,)
    
    # Sort by sum value
    order = np.argsort(flat_sums)
    sorted_sums = flat_sums[order]
    sorted_weights = flat_weights[order]
    cum_weights = np.cumsum(sorted_weights)
    
    # For each (k,l), find how many pair sums are < 2*s_l - s_k
    # thresholds[k,l] = 2*s_l - s_k
    thresholds = 2.0 * support[None, :] - support[:, None]  # (n, n): [k,l] -> 2*s_l - s_k
    
    # For each threshold, find count of sorted_sums < threshold
    # Use searchsorted
    counts = np.searchsorted(sorted_sums, thresholds, side='left')  # (n, n)
    
    # The cumulative weight up to index counts[k,l] (exclusive)
    # cum_weights[i] = sum of sorted_weights[0..i-1]
    # So we want cum_weights[counts[k,l]] but handle counts=0
    # Use: result = np.where(counts > 0, cum_weights[counts-1], 0)
    # But cum_weights has length n*n, and counts can be up to n*n
    # Actually cum_weights[i] for i in 0..n*n, where cum_weights[i] = sum of first i elements
    # Let me redefine: cum_weights_full = np.concatenate([[0], np.cumsum(sorted_weights)])
    cum_weights_full = np.concatenate([[0.0], np.cumsum(sorted_weights)])
    
    # Now for each (k,l), the mass is cum_weights_full[counts[k,l]]
    mass = cum_weights_full[counts]  # (n, n)
    
    # P = sum_{k,l} w_k * w_l * mass[k,l]
    P = np.sum(w[:, None] * w[None, :] * mass)
    
    return P


def _two_point_P(p):
    """Analytic P for 2-point measure on {0,1} with P(X=1)=p."""
    return p * (1-p)**2 * (1 + 2*p)


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Uses a 2-point measure on {0, 1} with optimized weight, plus attempts to improve
    with small supports using fast P computation.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    # Best known: 2-point on {0,1}
    # Optimize p in _two_point_P(p) = p(1-p)^2(1+2p)
    # f'(p) = 1 - 9p^2 + 8p^3 = 0
    # Roots: p=1, p=(1-sqrt(33))/16 < 0, p=(1+sqrt(33))/16
    p_opt = (1 + np.sqrt(33.0)) / 16.0
    p_opt = np.clip(p_opt, 1e-9, 1 - 1e-9)
    
    support_2 = np.array([0.0, 1.0], dtype=np.float64)
    weights_2 = np.array([1.0 - p_opt, p_opt], dtype=np.float64)
    P_best = _two_point_P(p_opt)
    best_support = support_2.copy()
    best_weights = weights_2.copy()
    
    # Try to improve with 3-point measures
    if time.time() < deadline - 100:
        # Try geometric-like supports
        # Support: {0, a, b} with weights {w0, wa, wb}
        # Try some structured supports
        
        # Try {0, 1, 2} with various weights
        support_3 = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        
        # Grid search over weights
        best_P_3 = P_best
        best_w_3 = None
        
        # Use a coarse grid first
        for w0 in np.linspace(0.1, 0.8, 10):
            for wa in np.linspace(0.05, 0.8 - w0, 10):
                wb = 1.0 - w0 - wa
                if wb < 0.01:
                    continue
                w = np.array([w0, wa, wb])
                P = _compute_P_fast(support_3, w)
                if P > best_P_3:
                    best_P_3 = P
                    best_w_3 = w.copy()
        
        if best_w_3 is not None and best_P_3 > P_best:
            P_best = best_P_3
            best_support = support_3.copy()
            best_weights = best_w_3.copy()
    
    # Try {0, 1, 3} and {0, 2, 4}
    if time.time() < deadline - 80:
        for test_support in [np.array([0.0, 1.0, 3.0]), np.array([0.0, 2.0, 4.0])]:
            best_P_s = P_best
            best_w_s = None
            
            for w0 in np.linspace(0.1, 0.8, 8):
                for ws in np.linspace(0.05, 0.8 - w0, 8):
                    wl = 1.0 - w0 - ws
                    if wl < 0.01:
                        continue
                    w = np.array([w0, ws, wl])
                    P = _compute_P_fast(test_support, w)
                    if P > best_P_s:
                        best_P_s = P
                        best_w_s = w.copy()
            
            if best_w_s is not None and best_P_s > P_best:
                P_best = best_P_s
                best_support = test_support.copy()
                best_weights = best_w_s.copy()
    
    # Try 4-point: {0, 1, 2, 3}
    if time.time() < deadline - 60:
        support_4 = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        best_P_4 = P_best
        best_w_4 = None
        
        # Coarse grid
        for w0 in np.linspace(0.1, 0.6, 6):
            for w1 in np.linspace(0.05, 0.6 - w0, 6):
                for w2 in np.linspace(0.05, 0.6 - w0 - w1, 6):
                    w3 = 1.0 - w0 - w1 - w2
                    if w3 < 0.01:
                        continue
                    w = np.array([w0, w1, w2, w3])
                    P = _compute_P_fast(support_4, w)
                    if P > best_P_4:
                        best_P_4 = P
                        best_w_4 = w.copy()
        
        if best_w_4 is not None and best_P_4 > P_best:
            P_best = best_P_4
            best_support = support_4.copy()
            best_weights = best_w_4.copy()
    
    # Validate result
    if len(best_support) != len(best_weights):
        # Fallback to 2-point
        best_support = support_2
        best_weights = weights_2
    
    if np.any(np.isnan(best_support)) or np.any(np.isinf(best_support)):
        best_support = support_2
        best_weights = weights_2
    
    if np.any(best_support < -1e-9):
        best_support = support_2
        best_weights = weights_2
    
    if np.any(best_weights < -1e-9):
        best_weights = weights_2
        best_support = support_2
    
    w_sum = np.sum(best_weights)
    if w_sum <= 1e-9:
        best_support = support_2
        best_weights = weights_2
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
