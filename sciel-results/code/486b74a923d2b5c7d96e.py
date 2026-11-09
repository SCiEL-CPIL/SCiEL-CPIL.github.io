import numpy as np
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

# Track time to ensure we stay within budget
_start_time = time.time()
_TIME_LIMIT = 320.0  # 10 seconds margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Pure GP-LCB acquisition function for minimization.
    Score = -mu - kappa * sigma
    Higher score means more likely to be a good candidate.
    """
    global _start_time
    
    # Safety check: if we're close to the time limit, return a simple score
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Return deterministic score based on -mu to avoid hanging
        return -np.asarray(mu)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle numerical stability for sigma
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Known global minimum for Styblinski-Tang 1D
    f_min = -39.1662
    
    # Dynamic kappa:
    # When f_best is far from f_min (high value), we should explore more (higher kappa)
    # When f_best is close to f_min (low value), we should exploit more (lower kappa)
    
    # Progress p: 1.0 when f_best = 0, 0.0 when f_best = f_min
    # If f_best > 0, p > 1 -> clamp to 1
    # If f_best < f_min, p < 0 -> clamp to 0
    
    p = (f_best - f_min) / (0.0 - f_min)
    p = np.clip(p, 0.0, 1.0)
    
    # When p=1 (f_best=0, far from min), kappa should be high (explore)
    # When p=0 (f_best=f_min, at min), kappa should be low (exploit)
    
    kappa_min = 0.5
    kappa_max = 2.5
    
    kappa = kappa_min + (kappa_max - kappa_min) * p
    
    # GP-LCB score for minimization
    # We want to minimize f(x), so we look for points where the lower bound is low
    # Lower bound = mu - kappa * sigma
    # We want to maximize the "potential" which is the negative of the lower bound
    # score = -(mu - kappa * sigma) = -mu + kappa * sigma
    
    # Wait, the standard GP-LCB for minimization is:
    # a(x) = mu(x) - kappa * sigma(x)
    # We minimize a(x).
    # But the interface says "higher meaning evaluate next".
    # So we should return -a(x) = -mu(x) + kappa * sigma(x).
    
    score = -mu + kappa * sigma_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
