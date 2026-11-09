# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly for a discrete measure."""
    n = len(support)
    w = weights / weights.sum()
    # Build pairwise sums matrix: S[i,j] = s_i + s_j
    S = support[:, None] + support[None, :]
    # For each (k, l): need sum over (i,j) where S[i,j] < 2*s_l - s_k
    # P = sum_k sum_l w_k w_l * (sum_{i,j: S[i,j] < 2*s_l - s_k} w_i w_j)
    # Vectorized: for each (k,l), count the weighted sum
    # 2*s_l - s_k for all (k,l)
    thresholds = 2.0 * support[None, :] - support[:, None]  # shape (n, n), index [k, l]
    # For each threshold, we need sum of w_i*w_j over S[i,j] < threshold
    # This is expensive for large n. Use n=20-40 range.
    # Build weight matrix W = w_i * w_j
    W = np.outer(w, w)
    P = 0.0
    for k in range(n):
        for l in range(n):
            thr = 2.0 * support[l] - support[k]
            mask = S < thr
            P += w[k] * w[l] * (W * mask).sum()
    return P


def _compute_P_fast(support, weights):
    """Faster computation using sorted pairwise sums."""
    n = len(support)
    w = weights / weights.sum()
    S = support[:, None] + support[None, :]
    W = np.outer(w, w)
    # Flatten S and W for sorting
    S_flat = S.ravel()
    W_flat = W.ravel()
    order = np.argsort(S_flat, kind='mergesort')
    S_sorted = S_flat[order]
    W_sorted = W_flat[order]
    W_cdf = np.cumsum(W_sorted)
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            thr = 2.0 * support[l] - support[k]
            # Number of pairs with S < thr
            idx = np.searchsorted(S_sorted, thr, side='left')
            if idx > 0:
                P += w[k] * w[l] * W_cdf[idx - 1]
            # else: 0
    return P


def _optimize_weights(support, max_time=300.0):
    """Optimize weights on a fixed support to maximize P."""
    n = len(support)
    deadline = time.time() + max_time
    
    # Start with uniform weights
    best_w = np.ones(n) / n
    best_P = _compute_P_fast(support, best_w)
    
    # Use a simple hill-climbing / Nelder-Mead-like approach
    # with random restarts
    from scipy.optimize import minimize
    
    def neg_P(w_raw):
        # Ensure non-negative and normalized
        w = np.abs(w_raw)
        s = w.sum()
        if s < 1e-15:
            return 1.0
        w = w / s
        return -_compute_P_fast(support, w)
    
    # Nelder-Mead on log-weights to keep them positive
    def neg_P_log(w_log):
        w = np.exp(w_log - w_log.max())
        s = w.sum()
        if s < 1e-15:
            return 1.0
        w = w / s
        return -_compute_P_fast(support, w)
    
    # Initial: uniform in log space
    x0 = np.zeros(n)
    
    # Run Nelder-Mead
    result = minimize(neg_P_log, x0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10,
                               'adaptive': True})
    
    w_opt = np.exp(result.x - result.x.max())
    w_opt = w_opt / w_opt.sum()
    P_opt = -result.fun
    
    if P_opt > best_P:
        best_P = P_opt
        best_w = w_opt
    
    # Try a few random restarts if time permits
    rng = np.random.default_rng(42)
    for _ in range(10):
        if time.time() > deadline - 60:
            break
        x0 = rng.normal(0, 1, n)
        result = minimize(neg_P_log, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10,
                                   'adaptive': True})
        w_opt = np.exp(result.x - result.x.max())
        w_opt = w_opt / w_opt.sum()
        P_opt = -result.fun
        if P_opt > best_P:
            best_P = P_opt
            best_w = w_opt
    
    return best_w, best_P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: Use a log-spaced (geometric) support to capture both small and large
    values, then optimize the weights via Nelder-Mead to maximize P.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights.
    """
    start_time = time.time()
    deadline = start_time + 570  # Safety margin: 600s limit, leave 30s
    
    # Try different support sizes and spacings
    best_support = np.arange(32, dtype=np.float64)
    best_weights = np.ones(32, dtype=np.float64) / 32
    best_P = _compute_P_fast(best_support, best_weights)
    
    # Log-spaced supports: geometric progression
    # s_i = a * r^i for i = 0..n-1
    configs = [
        # (n, a, r) -> support = a * r^arange(n)
        (16, 0.01, 2.0),
        (16, 0.01, 3.0),
        (20, 0.01, 2.0),
        (20, 0.01, 3.0),
        (24, 0.01, 2.0),
        (24, 0.01, 3.0),
        (28, 0.01, 2.0),
        (32, 0.01, 2.0),
        (32, 0.01, 3.0),
        (16, 0.1, 2.0),
        (20, 0.1, 2.0),
        (24, 0.1, 2.0),
        (20, 0.5, 2.0),
        (24, 0.5, 2.0),
        (16, 1.0, 2.0),
        (20, 1.0, 2.0),
        (24, 1.0, 2.0),
        (16, 0.001, 5.0),
        (20, 0.001, 5.0),
        (24, 0.001, 5.0),
        (28, 0.001, 5.0),
    ]
    
    for n, a, r in configs:
        if time.time() > deadline - 120:
            break
        
        support = a * (r ** np.arange(n, dtype=np.float64))
        # Ensure all support >= 0 (they should be since a > 0, r > 0)
        if np.any(support < 0) or not np.all(np.isfinite(support)):
            continue
        
        # Quick check with uniform weights
        w_uniform = np.ones(n) / n
        P_uniform = _compute_P_fast(support, w_uniform)
        
        # If even uniform is bad, skip optimization
        if P_uniform < best_P * 0.5:
            continue
        
        # Optimize weights
        w_opt, P_opt = _optimize_weights(support, max_time=180.0)
        
        if P_opt > best_P:
            best_P = P_opt
            best_support = support.copy()
            best_weights = w_opt.copy()
    
    # Final safety check
    if time.time() > deadline:
        # Return the original baseline if we're over time
        return np.arange(32, dtype=np.float64), np.ones(32, dtype=np.float64) / 32
    
    # Validate
    if len(best_support) != len(best_weights):
        return np.arange(32, dtype=np.float64), np.ones(32, dtype=np.float64) / 32
    if np.any(best_support < -1e-9) or np.any(best_weights < -1e-9):
        return np.arange(32, dtype=np.float64), np.ones(32, dtype=np.float64) / 32
    if not np.all(np.isfinite(best_support)) or not np.all(np.isfinite(best_weights)):
        return np.arange(32, dtype=np.float64), np.ones(32, dtype=np.float64) / 32
    if best_weights.sum() <= 1e-9:
        return np.arange(32, dtype=np.float64), np.ones(32, dtype=np.float64) / 32
    
    return best_support.astype(np.float64), best_weights.astype(np.float64)


# EVOLVE-BLOCK-END
