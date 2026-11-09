# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Deadline guard: the evaluator kills after 630s. We use 600s as a safety margin.
_DEADLINE = 630.0
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _remaining():
    return _DEADLINE - _elapsed() - 10.0  # 10s safety margin


def _calc_prob(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] for i.i.d. draws from the discrete measure.
    support: 1-D array of shape (n,)
    weights: 1-D array of shape (n,), will be normalized internally.
    """
    n = len(support)
    if n == 0:
        return 0.0
    w = weights / weights.sum()
    
    # Pairwise sums: s_i + s_j
    # We need sum over k, l of w_k * w_l * (sum of w_i * w_j over pairs (i, j) with s_i + s_j < 2*s_l - s_k)
    # Let A = s_i + s_j. We want P[A < 2*s_l - s_k].
    # Equivalent: P[s_i + s_j + s_k < 2*s_l].
    # Let Y = s_i + s_j + s_k. We want P[Y < 2*s_l].
    
    # Vectorized approach:
    # Build all pairwise sums s_i + s_j with weights w_i * w_j
    # Then for each k, l, check sum over (i,j) where s_i + s_j < 2*s_l - s_k
    
    # More efficient: 
    # For each l, threshold T_l = 2*s_l.
    # For each k, we need P[s_i + s_j < T_l - s_k].
    
    # Let's compute all pairwise sums and their weights.
    # n^2 pairs.
    si = support[:, None]
    sj = support[None, :]
    pair_sums = si + sj  # shape (n, n)
    pair_weights = w[:, None] * w[None, :]  # shape (n, n)
    
    # Flatten
    ps_flat = pair_sums.flatten()
    pw_flat = pair_weights.flatten()
    
    # Sort pair_sums for efficient cumulative sum
    idx = np.argsort(ps_flat)
    ps_sorted = ps_flat[idx]
    pw_sorted = pw_flat[idx]
    cw_sorted = np.cumsum(pw_sorted)
    
    total_prob = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * support[l] - support[k]
            # Strict inequality: s_i + s_j < threshold
            # Find rightmost index where ps_sorted < threshold
            # np.searchsorted with side='left' gives first index >= threshold
            # So indices [0, idx) are < threshold
            idx_right = np.searchsorted(ps_sorted, threshold, side='left')
            if idx_right > 0:
                total_prob += w[k] * w[l] * cw_sorted[idx_right - 1]
            # else: 0
    
    return total_prob


def _calc_prob_fast(support, weights):
    """
    Optimized computation using sorting and cumulative sums.
    """
    n = len(support)
    if n == 0:
        return 0.0
    w = weights / weights.sum()
    
    # Build all pairwise sums
    si = support[:, None]
    sj = support[None, :]
    pair_sums = si + sj
    pair_weights = w[:, None] * w[None, :]
    
    ps_flat = pair_sums.flatten()
    pw_flat = pair_weights.flatten()
    
    idx = np.argsort(ps_flat)
    ps_sorted = ps_flat[idx]
    pw_sorted = pw_flat[idx]
    cw_sorted = np.cumsum(pw_sorted)
    
    # For each (k, l) pair, threshold = 2*s_l - s_k
    # We can vectorize over l for each k
    total_prob = 0.0
    for k in range(n):
        thresholds = 2.0 * support - support[k]  # shape (n,)
        # For each threshold, find count of pair_sums < threshold
        idx_rights = np.searchsorted(ps_sorted, thresholds, side='left')
        # For each l, the cumulative weight is cw_sorted[idx_right - 1] if idx_right > 0 else 0
        # Handle idx_right = 0
        idx_safe = np.maximum(idx_rights - 1, 0)
        cum_weights = cw_sorted[idx_safe]
        # Where idx_right == 0, cum_weights should be 0
        cum_weights[idx_rights == 0] = 0.0
        total_prob += w[k] * np.sum(w * cum_weights)
    
    return total_prob


def _optimize_3point():
    """
    Optimize a 3-point distribution: support {0, 1, M} with weights (p, q, r).
    p + q + r = 1, p, q, r > 0, M > 0.
    """
    def _prob_3point(params):
        p, q, M = params
        r = 1.0 - p - q
        if p < 0 or q < 0 or r < 0 or M <= 0:
            return -1.0
        support = np.array([0.0, 1.0, M])
        weights = np.array([p, q, r])
        return _calc_prob_fast(support, weights)
    
    def _neg_prob(params):
        prob = _prob_3point(params)
        if prob < 0:
            return 1.0
        return -prob
    
    # Try multiple starting points
    best_prob = 0.0
    best_params = None
    
    start_points = [
        [0.3, 0.3, 10.0],
        [0.2, 0.2, 5.0],
        [0.1, 0.1, 2.0],
        [0.4, 0.4, 20.0],
        [0.5, 0.1, 1.5],
        [0.1, 0.5, 3.0],
        [0.2, 0.6, 1.2],
        [0.6, 0.2, 8.0],
    ]
    
    for sp in start_points:
        if _remaining() < 5.0:
            break
        try:
            res = minimize(_neg_prob, sp, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < 0:
                prob = -res.fun
                if prob > best_prob:
                    best_prob = prob
                    best_params = res.x
        except Exception:
            pass
    
    if best_params is not None:
        p, q, M = best_params
        r = 1.0 - p - q
        if p >= 0 and q >= 0 and r >= 0 and M > 0:
            support = np.array([0.0, 1.0, M])
            weights = np.array([p, q, r])
            return support, weights, best_prob
    
    return None, None, 0.0


def _optimize_4point():
    """
    Optimize a 4-point distribution: support {0, 1, M1, M2} with weights.
    """
    def _prob_4point(params):
        p, q, r, M1, M2 = params
        s = 1.0 - p - q - r
        if p < 0 or q < 0 or r < 0 or s < 0 or M1 <= 0 or M2 <= 0 or M2 <= M1:
            return -1.0
        support = np.array([0.0, 1.0, M1, M2])
        weights = np.array([p, q, r, s])
        return _calc_prob_fast(support, weights)
    
    def _neg_prob(params):
        prob = _prob_4point(params)
        if prob < 0:
            return 1.0
        return -prob
    
    best_prob = 0.0
    best_params = None
    
    start_points = [
        [0.2, 0.2, 0.2, 5.0, 20.0],
        [0.1, 0.1, 0.1, 3.0, 10.0],
        [0.3, 0.3, 0.1, 2.0, 8.0],
        [0.15, 0.15, 0.15, 4.0, 15.0],
    ]
    
    for sp in start_points:
        if _remaining() < 5.0:
            break
        try:
            res = minimize(_neg_prob, sp, method='Nelder-Mead',
                          options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < 0:
                prob = -res.fun
                if prob > best_prob:
                    best_prob = prob
                    best_params = res.x
        except Exception:
            pass
    
    if best_params is not None:
        p, q, r, M1, M2 = best_params
        s = 1.0 - p - q - r
        if p >= 0 and q >= 0 and r >= 0 and s >= 0 and M1 > 0 and M2 > M1:
            support = np.array([0.0, 1.0, M1, M2])
            weights = np.array([p, q, r, s])
            return support, weights, best_prob
    
    return None, None, 0.0


def _baseline_uniform():
    """Baseline: uniform on {0, 1, ..., 31}."""
    n = 32
    support = np.arange(n, dtype=np.float64)
    weights = np.ones(n, dtype=np.float64) / n
    prob = _calc_prob_fast(support, weights)
    return support, weights, prob


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    _START_TIME_LOCAL = time.time()
    DEADLINE_LOCAL = _START_TIME_LOCAL + 620.0  # 10s safety margin
    
    best_support = None
    best_weights = None
    best_prob = 0.0
    
    # 1. Baseline
    support, weights, prob = _baseline_uniform()
    if prob > best_prob:
        best_support, best_weights, best_prob = support, weights, prob
    
    # 2. Try 3-point optimization
    if time.time() < DEADLINE_LOCAL - 10.0:
        s3, w3, p3 = _optimize_3point()
        if s3 is not None and p3 > best_prob:
            best_support, best_weights, best_prob = s3, w3, p3
    
    # 3. Try 4-point optimization
    if time.time() < DEADLINE_LOCAL - 10.0:
        s4, w4, p4 = _optimize_4point()
        if s4 is not None and p4 > best_prob:
            best_support, best_weights, best_prob = s4, w4, p4
    
    # 4. Try a geometric-like distribution
    if time.time() < DEADLINE_LOCAL - 10.0:
        for ratio in [2.0, 3.0, 4.0, 5.0]:
            n = 20
            support = ratio ** np.arange(n)
            # Weights: try various decays
            for decay in [0.5, 0.7, 0.8, 0.9]:
                weights = decay ** np.arange(n)
                weights /= weights.sum()
                prob = _calc_prob_fast(support, weights)
                if prob > best_prob:
                    best_support, best_weights, best_prob = support, weights, prob
    
    if best_support is None:
        # Fallback
        n = 32
        best_support = np.arange(n, dtype=np.float64)
        best_weights = np.ones(n, dtype=np.float64) / n
    
    # Ensure validity
    best_support = best_support.astype(np.float64)
    best_weights = best_weights.astype(np.float64)
    
    # Clip small negative values
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    # Ensure weights sum > 1e-9
    if best_weights.sum() <= 1e-9:
        best_weights = np.ones_like(best_weights) / len(best_weights)
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
