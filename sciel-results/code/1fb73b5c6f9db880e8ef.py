# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_prob(r1, r2, p, n=256):
    """
    Compute P[X1+X2+X3 < 2*X4] for a mixture of two geometric distributions.
    r1, r2: decay parameters (0 < r < 1)
    p: weight for component 1
    n: number of support points
    """
    support = np.arange(n, dtype=np.float64)
    
    # Geometric PMF: (1-r) * r^i
    # Ensure numerical stability
    # If r is very close to 1, r**i decays slowly.
    # If r is very close to 0, r**i decays fast.
    
    w1 = (1 - r1) * (r1 ** support)
    w2 = (1 - r2) * (r2 ** support)
    
    weights = p * w1 + (1 - p) * w2
    
    # Normalize just in case
    total_w = np.sum(weights)
    if total_w <= 1e-9:
        return 0.0
    weights = weights / total_w
    
    # Compute CDF of S = X1 + X2
    # S takes values in [0, 2*(n-1)]
    # We need P[S < 2*X4 - X3]
    
    # Create a histogram of sums
    i = np.arange(n)
    j = np.arange(n)
    sums = i[:, None] + j[None, :]
    probs = weights[:, None] * weights[None, :]
    
    # sums is int array, probs is float
    # bincount requires non-negative ints
    sum_probs = np.bincount(sums.flatten(), weights=probs.flatten(), minlength=2*n-1)
    
    # CDF of S: CDF[t] = P[S <= t]
    cdf = np.cumsum(sum_probs)
    
    # Now sum over k, l
    # term = w_k * w_l * P[S < 2*s_l - s_k]
    # Let x = 2*s_l - s_k
    # If x <= 0, P[S < x] = 0
    # If x >= 2n-1, P[S < x] = 1
    
    k = np.arange(n)
    l = np.arange(n)
    x = 2 * l[:, None] - k[None, :] # Shape (n, n)
    
    indices = x - 1
    
    # Initialize result matrix
    prob_s_less_x = np.zeros((n, n), dtype=np.float64)
    
    # Mask for valid indices (0 <= index <= 2n-2)
    mask = (indices >= 0) & (indices < 2*n-1)
    prob_s_less_x[mask] = cdf[indices[mask]]
    
    # Mask for x > 2n-1 (i.e. index >= 2n-1)
    mask_high = indices >= 2*n-1
    prob_s_less_x[mask_high] = 1.0
    
    # Final sum
    # Sum over k, l of w_k * w_l * prob_s_less_x[l, k]
    w_outer = weights[None, :] * weights[:, None] # Shape (n, n), [l, k]
    
    p_val = np.sum(w_outer * prob_s_less_x)
    
    return p_val

def _objective(params):
    # params are log-odds or log-values to ensure constraints
    # Let's use logit for p (0 to 1) and log for r (0 to 1)
    # Actually, simple log for r is fine if we clip, but let's use a transformation that maps R to (0,1).
    # r = 1 / (1 + exp(-x)) ? No, let's stick to simple bounds in optimizer but use log-space for the decay rate to allow wider range.
    # Let log_r1, log_r2 be such that r = exp(log_r). We need r < 1, so log_r < 0.
    # Let's optimize over t1, t2 where r = 1 - exp(-exp(t))? No, simpler:
    # r = 1 / (1 + exp(t)) ? No.
    
    # Let's just use the raw r values but clip them in the objective, and let the optimizer handle it.
    # The previous program did this. The directive suggests "log-space variables".
    # Let's define r = exp(log_r). We constrain log_r < 0.
    # And p = sigmoid(log_p).
    
    log_r1, log_r2, log_p = params
    
    r1 = np.exp(log_r1)
    r2 = np.exp(log_r2)
    p = 1.0 / (1.0 + np.exp(-log_p))
    
    # Clip to valid ranges to avoid numerical issues
    # r must be in (0, 1)
    r1 = np.clip(r1, 1e-6, 1.0 - 1e-6)
    r2 = np.clip(r2, 1e-6, 1.0 - 1e-6)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    
    prob = _compute_prob(r1, r2, p)
    return -prob # Minimize negative probability

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    n = 256
    support = np.arange(n, dtype=np.float64)
    
    # Initial guess in log-space
    # r ~ 0.9 => log_r ~ -0.1
    # r ~ 0.5 => log_r ~ -0.69
    # p ~ 0.5 => log_p ~ 0
    x0 = np.array([-0.1, -0.7, 0.0])
    
    # Optimize
    # Use Nelder-Mead for robustness without gradients
    res = minimize(_objective, x0, method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
    
    log_r1, log_r2, log_p = res.x
    
    r1 = np.exp(log_r1)
    r2 = np.exp(log_r2)
    p = 1.0 / (1.0 + np.exp(-log_p))
    
    # Final clipping
    r1 = np.clip(r1, 1e-6, 1.0 - 1e-6)
    r2 = np.clip(r2, 1e-6, 1.0 - 1e-6)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    
    w1 = (1 - r1) * (r1 ** support)
    w2 = (1 - r2) * (r2 ** support)
    weights = p * w1 + (1 - p) * w2
    
    # Normalize
    weights = weights / np.sum(weights)
    
    return support, weights
# EVOLVE-BLOCK-END
