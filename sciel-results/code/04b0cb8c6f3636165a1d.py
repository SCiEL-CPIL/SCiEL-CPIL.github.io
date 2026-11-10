# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _compute_p(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly."""
    n = len(support)
    w = weights / weights.sum()
    
    # Sort support and weights
    order = np.argsort(support)
    s_sorted = support[order]
    w_sorted = w[order]
    
    # Precompute cumulative weights for efficient prefix sums
    cum_w = np.cumsum(w_sorted)
    cum_w = np.concatenate([[0.0], cum_w])
    
    p_total = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2 * s_sorted[l] - s_sorted[k]
            if threshold <= 0:
                continue
            
            # Sum over i of w_i * (sum over j with s_j < threshold - s_i of w_j)
            inner = 0.0
            for i in range(n):
                # Find max index j such that s_sorted[j] < threshold - s_sorted[i]
                max_j = np.searchsorted(s_sorted, threshold - s_sorted[i], side='left')
                if max_j > 0:
                    inner += w_sorted[i] * cum_w[max_j]
            
            p_total += w_sorted[k] * w_sorted[l] * inner
    
    return p_total


def _neg_p(params, n_points):
    """
    Objective function to minimize (negative probability).
    params: [log_base, log_ratio, log_w_0, ..., log_w_{n-1}]
    Support: s_i = exp(log_base + i * log_ratio)
    Weights: w_i = exp(log_w_i)
    """
    log_base = params[0]
    log_ratio = params[1]
    log_ws = params[2:]
    
    i = np.arange(n_points)
    log_support = log_base + i * log_ratio
    support = np.exp(log_support)
    
    weights = np.exp(log_ws)
    
    # Check for validity
    if not np.all(np.isfinite(support)) or not np.all(np.isfinite(weights)):
        return 1e10
    if np.any(support < -1e-9) or np.any(weights < -1e-9):
        return 1e10
    if np.sum(weights) <= 1e-9:
        return 1e10
        
    p = _compute_p(support, weights)
    return -p


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1+X2+X3 < 2*X4].
    
    Approach: Use a geometrically spaced support and optimized weights.
    This allows the distribution to adapt its scale and "tail" shape continuously.
    """
    n_points = 12
    
    # Initial guess
    # Support roughly geometric from 0.1 to 100
    # log_base ~ log(0.1) ~ -2.3
    # log_ratio ~ log(10^(2/11)) ~ 0.45
    log_base_init = np.log(0.1)
    log_ratio_init = 0.5
    
    # Weights: Start with a bias towards the tail or bulk?
    # Let's start with slightly decaying weights to mimic a tail
    log_ws_init = np.linspace(0, -1, n_points)
    
    x0 = np.concatenate([[log_base_init], [log_ratio_init], log_ws_init])
    
    # Use Nelder-Mead with adaptive settings
    result = minimize(
        _neg_p,
        x0,
        args=(n_points,),
        method='Nelder-Mead',
        options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
    )
    
    x_opt = result.x
    log_base = x_opt[0]
    log_ratio = x_opt[1]
    log_ws = x_opt[2:]
    
    i = np.arange(n_points)
    support = np.exp(log_base + i * log_ratio)
    weights = np.exp(log_ws)
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Ensure valid output
    if not np.all(np.isfinite(support)) or not np.all(np.isfinite(weights)):
        # Fallback to a simple valid measure
        support = np.array([0.0, 1.0, 2.0, 10.0])
        weights = np.array([0.5, 0.3, 0.1, 0.1])
    
    return support, weights


# EVOLVE-BLOCK-END
