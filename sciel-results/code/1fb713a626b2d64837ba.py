"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guarding
_START_TIME = time.time()
# Safety margin in seconds (300s limit, leave 10s buffer)
_DEADLINE_MARGIN = 290.0


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization (minimization).
    Uses Expected Improvement (EI) which is deterministic and stable for small budgets.
    """
    # Deadline guard: if we are too close to the time limit, return a safe, fast result
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_MARGIN:
        # Fallback to a simple exploitation score to ensure we return quickly
        # This case is highly unlikely given the vectorized nature of the operation
        return -mu
    
    # Ensure sigma is non-negative and finite to avoid division by zero or negative std
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # We maximize EI, so higher scores are better.
    
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values in CDF/PDF
    # norm.cdf(-inf) = 0, norm.cdf(inf) = 1
    # norm.pdf is 0 at extremes
    z_clipped = np.clip(z, -10, 10)
    
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    
    scores = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
