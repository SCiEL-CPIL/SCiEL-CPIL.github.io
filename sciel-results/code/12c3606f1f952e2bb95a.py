# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from collections import defaultdict

def _entropy_from_values(values, weights):
    """Compute Shannon entropy in bits given values and their weights."""
    d = defaultdict(float)
    for v, w in zip(values, weights):
        d[v] += w
    total = sum(d.values())
    if total <= 1e-15:
        return 0.0
    probs = [w / total for w in d.values() if w > 1e-15]
    if not probs:
        return 0.0
    return -sum(p * np.log2(p) for p in probs)

def _compute_ratio(P):
    """Compute D/M ratio for a 10x10 distribution P[y,x]."""
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return 0.0
    P = P / s
    
    y_idx, x_idx = np.indices((10, 10))
    x_flat = x_idx.flatten()
    y_flat = y_idx.flatten()
    w = P.flatten()
    
    # D = H of (x - y)
    d_vals = x_flat - y_flat
    D = _entropy_from_values(d_vals, w)
    
    # M = max(H_0, H_1, H_2)
    h0 = _entropy_from_values(x_flat, w)  # x
    h1 = _entropy_from_values(x_flat + y_flat, w)  # x + y
    h2 = _entropy_from_values(x_flat + 2 * y_flat, w)  # x + 2y
    M = max(h0, h1, h2)
    
    if M <= 1e-9:
        return 0.0
    
    return D / M

def _neg_ratio(params):
    """Negative ratio for minimization."""
    P = np.exp(params).reshape(10, 10)
    return -_compute_ratio(P)

def _optimize_P(start_logP, deadline, max_iter=200):
    """Optimize using Nelder-Mead on log-probability parameters."""
    start_time = time.time()
    
    # Initial guess
    x0 = start_logP.flatten().copy()
    
    # Nelder-Mead options
    options = {
        'maxiter': max_iter,
        'xatol': 1e-8,
        'fatol': 1e-8,
        'adaptive': True
    }
    
    result = minimize(_neg_ratio, x0, method='Nelder-Mead', options=options)
    
    # Check deadline
    if time.time() > deadline:
        return None
    
    best_params = result.x
    best_P = np.exp(best_params).reshape(10, 10)
    
    # Normalize and clip
    best_P = np.clip(best_P, 0, None)
    total = best_P.sum()
    if total > 1e-15:
        best_P = best_P / total
    
    return best_P

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 600  # 600 second deadline
    
    # Start with uniform distribution
    P_uniform = np.ones((n, n), dtype=np.float64) / (n * n)
    best_P = P_uniform.copy()
    best_ratio = _compute_ratio(P_uniform)
    
    # Generate several random initial points
    np.random.seed(42)
    num_restarts = 20
    
    for restart in range(num_restarts):
        # Check deadline
        if time.time() > deadline - 30:  # Leave 30 seconds safety margin
            break
        
        # Generate random initial distribution
        # Use a mixture of uniform and some structured patterns
        if restart == 0:
            # Start near uniform
            logP = np.log(P_uniform) + np.random.normal(0, 0.1, (n, n))
        elif restart % 3 == 1:
            # Emphasize diagonal (x = y)
            P_init = np.zeros((n, n))
            for i in range(n):
                P_init[i, i] = 1.0
            P_init /= P_init.sum()
            logP = np.log(P_init + 1e-10) + np.random.normal(0, 0.5, (n, n))
        elif restart % 3 == 2:
            # Emphasize anti-diagonal (x + y = constant)
            P_init = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i + j == n - 1:
                        P_init[i, j] = 1.0
            P_init /= P_init.sum()
            logP = np.log(P_init + 1e-10) + np.random.normal(0, 0.5, (n, n))
        else:
            # Random distribution
            P_init = np.random.dirichlet(np.ones(n * n))
            P_init = P_init.reshape(n, n)
            logP = np.log(P_init + 1e-10)
        
        # Optimize
        result_P = _optimize_P(logP, deadline, max_iter=300)
        
        if result_P is not None:
            ratio = _compute_ratio(result_P)
            if ratio > best_ratio:
                best_ratio = ratio
                best_P = result_P.copy()
        
        # Check deadline
        if time.time() > deadline - 30:
            break
    
    # Final validation
    best_P = np.clip(best_P, 0, None)
    total = best_P.sum()
    if total <= 1e-9:
        return P_uniform
    best_P = best_P / total
    
    # Ensure no NaN or inf
    if not np.all(np.isfinite(best_P)):
        return P_uniform
    
    return best_P

# EVOLVE-BLOCK-END
