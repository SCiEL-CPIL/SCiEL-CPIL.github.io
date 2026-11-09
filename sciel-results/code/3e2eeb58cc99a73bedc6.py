# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_P(support, weights):
    """
    Compute P[X1 + X2 + X3 < 2*X4] exactly in float64.
    """
    n = len(support)
    if n == 0:
        return 0.0
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    s = support.astype(np.float64)
    
    # Pair sums and pair probabilities
    s_col = s.reshape(-1, 1)
    s_row = s.reshape(1, -1)
    pair_sums = s_col + s_row  # shape (n, n)
    pair_probs = np.outer(w, w)  # shape (n, n), pair_probs[i, j] = w_i * w_j
    
    # For each k, l: threshold = 2*s_l - s_k
    # We need sum_{i,j} pair_probs[i, j] * I(pair_sums[i, j] < threshold)
    # Then sum over k, l: w_k * w_l * (above)
    
    # This is a 4-index sum: sum_{k,l,i,j} w_k w_l w_i w_j I(s_i + s_j < 2*s_l - s_k)
    # Vectorize over (k, l) pairs.
    s_k = s.reshape(-1, 1, 1, 1)  # (n, 1, 1, 1)
    s_l = s.reshape(1, -1, 1, 1)  # (1, n, 1, 1)
    s_i = s.reshape(1, 1, -1, 1)  # (1, 1, n, 1)
    s_j = s.reshape(1, 1, 1, -1)  # (1, 1, 1, n)
    
    w_k = w.reshape(-1, 1, 1, 1)
    w_l = w.reshape(1, -1, 1, 1)
    w_i = w.reshape(1, 1, -1, 1)
    w_j = w.reshape(1, 1, 1, -1)
    
    # Condition: s_i + s_j < 2*s_l - s_k
    lhs = s_i + s_j
    rhs = 2.0 * s_l - s_k
    
    indicator = (lhs < rhs).astype(np.float64)
    
    weight_prod = w_k * w_l * w_i * w_j
    P = np.sum(indicator * weight_prod)
    return float(P)


def _optimize_weights_3point(support_fixed, deadline):
    """
    Optimize weights for a fixed 3-point support.
    """
    from scipy.optimize import minimize
    
    n = len(support_fixed)
    
    def neg_P(x):
        # x are log-weights for softmax
        exp_x = np.exp(x - np.max(x))
        w = exp_x / np.sum(exp_x)
        return -_compute_P(support_fixed, w)
    
    # Initial guess: equal weights
    x0 = np.zeros(n)
    
    # Try a few random starts
    best_P = -1.0
    best_w = None
    
    n_restarts = 10
    t_start = time.time()
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        if r == 0:
            x_init = x0
        else:
            x_init = np.random.randn(n) * 0.5
        
        try:
            res = minimize(neg_P, x_init, method='Nelder-Mead',
                           options={'maxiter': 5000, 'fatol': 1e-10, 'xatol': 1e-10})
            w = np.exp(res.x - np.max(res.x))
            w = w / np.sum(w)
            P_val = _compute_P(support_fixed, w)
            if P_val > best_P:
                best_P = P_val
                best_w = w
        except Exception:
            pass
    
    return best_w, best_P


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    t_start = time.time()
    deadline = t_start + 600.0  # Safety margin before 630s kill
    
    # Start with the seed baseline as fallback
    n_seed = 32
    support_seed = np.arange(n_seed, dtype=np.float64)
    weights_seed = np.ones(n_seed, dtype=np.float64) / n_seed
    best_P = _compute_P(support_seed, weights_seed)
    best_support = support_seed.copy()
    best_weights = weights_seed.copy()
    
    # Strategy 1: 3-point support with optimized weights
    # Try a few different support configurations
    support_configs = [
        np.array([0.0, 1.0, 10.0]),
        np.array([0.0, 1.0, 50.0]),
        np.array([0.0, 1.0, 100.0]),
        np.array([0.0, 1.0, 500.0]),
        np.array([0.0, 0.5, 10.0]),
        np.array([0.0, 0.5, 50.0]),
        np.array([0.0, 0.1, 10.0]),
        np.array([0.0, 0.1, 50.0]),
    ]
    
    for support in support_configs:
        if time.time() > deadline - 10:
            break
        w_opt, P_val = _optimize_weights_3point(support, deadline - 10)
        if P_val > best_P:
            best_P = P_val
            best_support = support.copy()
            best_weights = w_opt.copy()
    
    # Strategy 2: 4-point support with optimized weights
    support_configs_4 = [
        np.array([0.0, 1.0, 2.0, 50.0]),
        np.array([0.0, 1.0, 5.0, 100.0]),
        np.array([0.0, 0.5, 1.0, 50.0]),
    ]
    
    for support in support_configs_4:
        if time.time() > deadline - 10:
            break
        w_opt, P_val = _optimize_weights_3point(support, deadline - 10)
        if P_val > best_P:
            best_P = P_val
            best_support = support.copy()
            best_weights = w_opt.copy()
    
    # Ensure validity
    if len(best_support) != len(best_weights) or len(best_support) < 1:
        return support_seed, weights_seed
    
    # Clamp small negatives
    best_weights = np.maximum(best_weights, 0.0)
    best_support = np.maximum(best_support, 0.0)
    
    w_sum = np.sum(best_weights)
    if w_sum <= 1e-9:
        return support_seed, weights_seed
    
    return best_support, best_weights


# EVOLVE-BLOCK-END
