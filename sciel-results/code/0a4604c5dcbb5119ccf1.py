# EVOLVE-BLOCK-START
import numpy as np
import time
import random
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
    S_flat = S.flatten()
    
    # Weighted pair probabilities
    w_pair = w[:, None] * w[None, :]
    w_pair_flat = w_pair.flatten()
    
    # T[k, l] = 2*s_l - s_k
    T = 2.0 * support[None, :] - support[:, None]
    
    P = 0.0
    for k in range(n):
        for l in range(n):
            target = T[k, l]
            # Strict inequality
            mask = S_flat < target
            prob_kl = np.dot(w_pair_flat, mask)
            P += w[k] * w[l] * prob_kl
            
    return P

def _objective(params, n):
    """
    Objective function for Nelder-Mead: minimize -P.
    params: flat array of length 2n: first n are log-support, next n are log-weights.
    """
    log_s = params[:n]
    log_w = params[n:]
    
    support = np.exp(log_s)
    # Softmax for weights
    max_log_w = np.max(log_w)
    w_unnorm = np.exp(log_w - max_log_w)
    weights = w_unnorm / np.sum(w_unnorm)
    
    P = _evaluate_measure(support, weights)
    return -P

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Optimize the discrete measure using Nelder-Mead optimization with multiple restarts.
    """
    start_time = time.time()
    deadline = start_time + 585  # Safety margin: 15s buffer before 600s kill
    
    best_score = -np.inf
    best_support = None
    best_weights = None
    
    random.seed(42)
    np.random.seed(42)
    
    # Try different numbers of points
    candidate_ns = [3, 4, 5, 6]
    
    for n in candidate_ns:
        # Multiple restarts for each n
        num_restarts = 30
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Initialize support and weights
            # Use the 3-point insight: {0, 1, large} pattern for n=3
            if n == 3 and r < 10:
                # Start with the known good pattern
                support = np.array([0.0, 1.0, 10.0])
            elif r % 3 == 0:
                # Structured start
                support = np.arange(n, dtype=np.float64)
            elif r % 3 == 1:
                # Random start in [0, 5]
                support = np.sort(np.random.uniform(0, 5, n))
            else:
                # Random start with one large value
                support = np.sort(np.random.uniform(0, 3, n - 1))
                support = np.concatenate([support, [np.random.uniform(5, 15)]])
            
            # Weights: random positive, normalized
            weights = np.random.dirichlet(np.ones(n))
            
            # Convert to log-space for optimization
            log_s = np.log(support + 1e-15)
            log_w = np.log(weights + 1e-15)
            
            # Initial objective
            current_score = _evaluate_measure(support, weights)
            if current_score > best_score:
                best_score = current_score
                best_support = support.copy()
                best_weights = weights.copy()
            
            # Run Nelder-Mead optimization
            try:
                result = minimize(
                    _objective,
                    x0=np.concatenate([log_s, log_w]),
                    args=(n,),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                
                if result.success or result.fun < -current_score:
                    opt_params = result.x
                    opt_log_s = opt_params[:n]
                    opt_log_w = opt_params[n:]
                    
                    opt_support = np.exp(opt_log_s)
                    max_log_w = np.max(opt_log_w)
                    opt_w_unnorm = np.exp(opt_log_w - max_log_w)
                    opt_weights = opt_w_unnorm / np.sum(opt_w_unnorm)
                    
                    opt_score = -result.fun
                    if opt_score > current_score:
                        current_score = opt_score
                        support = opt_support
                        weights = opt_weights
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_support = support.copy()
                            best_weights = weights.copy()
            except Exception:
                pass
            
            # Check time
            if time.time() > deadline:
                break
    
    # If we didn't find anything (shouldn't happen), fallback
    if best_support is None:
        n = 3
        best_support = np.array([0.0, 1.0, 10.0])
        best_weights = np.array([0.3, 0.4, 0.3])
    
    # Final validity check
    if np.any(best_support < -1e-9) or np.any(best_weights < -1e-9):
        best_support = np.array([0.0, 1.0, 10.0])
        best_weights = np.array([0.3, 0.4, 0.3])
    
    return best_support, best_weights
# EVOLVE-BLOCK-END
