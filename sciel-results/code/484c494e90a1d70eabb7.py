"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard: kill if we exceed ~325s total (safety margin for 330s limit)
_DEADLINE = time.time() + 325.0


def acquisition(mu, sigma, f_best):
    """
    Probability of Improvement (PI) with a small exploration jitter.
    
    PI(x) = P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
    
    We add a small epsilon to sigma to avoid division by zero and to
    encourage slight exploration in areas of very low uncertainty.
    """
    # Deadline guard: if we're running out of time, return a safe fallback
    # that still produces finite scores. We'll use a simple heuristic:
    # prefer lower mu (exploitation) with a tiny bit of sigma (exploration).
    if time.time() > _DEADLINE - 5.0:
        # Fallback: score = -mu + 0.01 * sigma (exploit with slight explore)
        # This is finite and higher for lower mu.
        return -mu + 0.01 * sigma
    
    eps = 1e-4  # small exploration jitter
    sigma_safe = sigma + eps
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    # Higher PI means higher chance of beating f_best
    pi_scores = norm.cdf(z)
    
    # Ensure all values are finite
    pi_scores = np.where(np.isfinite(pi_scores), pi_scores, 0.0)
    
    return pi_scores
# EVOLVE-BLOCK-END
