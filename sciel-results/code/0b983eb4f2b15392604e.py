# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] for discrete measure."""
    n = len(support)
    if n == 0:
        return 0.0
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    
    # Build pairwise sums s_i + s_j
    s_col = support.reshape(-1, 1)
    s_row = support.reshape(1, -1)
    pair_sums = s_col + s_row
    pair_weights = np.outer(w, w)
    
    # For each k, compute sum_{i,j} w_i w_j I(s_i + s_j < 2*s_k)
    # Then P = sum_k w_k * (above)
    P = 0.0
    for k in range(n):
        threshold = 2.0 * support[k]
        mask = pair_sums < threshold
        P += w[k] * np.sum(pair_weights * mask)
    return P


def _decode(x):
    """Decode parameters to support and weights."""
    n = len(x) // 2
    s = x[:n]
    log_w = x[n:]
    
    # Ensure non-negative support
    s = np.maximum(s, 0.0)
    
    # Softplus for weights to ensure positivity
    w = np.exp(np.logaddexp(0, log_w))
    
    return s, w


def _negP(x):
    s, w = _decode(x)
    return -_compute_P(s, w)


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Start with a reasonable initial guess
    # Use points spread out: 0, small, medium, large
    n = 8
    init_support = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    init_weights = np.array([0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    # Convert to log-space for optimization
    init_log_w = np.log(np.maximum(init_weights, 1e-10))
    x0 = np.concatenate([init_support, init_log_w])
    
    best_x = x0.copy()
    best_P = _compute_P(*_decode(best_x))
    
    # Try optimization with time budget
    max_iter = 500
    try:
        def callback(xk):
            nonlocal best_x, best_P
            s, w = _decode(xk)
            P_val = _compute_P(s, w)
            if P_val > best_P:
                best_P = P_val
                best_x = xk.copy()
            return time.time() > deadline - 30.0
        
        res = minimize(_negP, x0, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'callback': callback,
                               'ftol': 1e-8, 'gtol': 1e-6})
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 60.0:
        s, w = _decode(best_x)
        # Ensure validity
        s = np.maximum(s, 0.0)
        w = np.maximum(w, 0.0)
        if np.sum(w) <= 1e-9:
            s = np.array([0.0])
            w = np.array([1.0])
        return s, w
    
    # Try a few more local optimizations from different starting points
    for trial in range(3):
        if time.time() > deadline - 120.0:
            break
        
        # Perturb best solution
        np.random.seed(trial)
        x_trial = best_x + np.random.randn(len(best_x)) * 0.5
        try:
            res2 = minimize(_negP, x_trial, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            s_trial, w_trial = _decode(res2.x)
            P_trial = _compute_P(s_trial, w_trial)
            if P_trial > best_P:
                best_P = P_trial
                best_x = res2.x.copy()
        except Exception:
            continue
    
    s, w = _decode(best_x)
    # Ensure validity
    s = np.maximum(s, 0.0)
    w = np.maximum(w, 0.0)
    if np.sum(w) <= 1e-9:
        s = np.array([0.0])
        w = np.array([1.0])
    
    return s, w


# EVOLVE-BLOCK-END
