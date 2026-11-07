# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global timer for safety
_t0 = time.time()
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - _t0 > _DEADLINE:
        # Return a score that favors low mu (exploitation) if we're out of time
        return -mu

    sigma_safe = np.maximum(sigma, 1e-8)

    # Probability of Improvement (PI) acquisition function
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This is a deterministic function that scores each candidate by
    # the probability of improving upon the current best.
    
    # Compute z-scores for the normalized improvement
    z = (f_best - mu) / sigma_safe
    
    # Compute the CDF of the standard normal distribution at z
    score = norm.cdf(z)
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
