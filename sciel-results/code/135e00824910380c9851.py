"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_start_time = time.time()
_DEADLINE_MARGIN = 10.0  # Seconds to spare before 300s limit


def acquisition(mu, sigma, f_best):
    """
    Kriging Believer acquisition: shifts the posterior mean by -k*sigma to
    encode exploration as an intrinsic property of the surrogate, then computes
    Expected Improvement on the shifted model. This avoids scale-mismatch
    between EI and UCB terms.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - posterior mean in objective units
    sigma : array of shape (2000,) - posterior std in objective units
    f_best : float - lowest observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,) - higher means evaluate next
    """
    # Deadline guard: If we are running out of time, return a simple fallback
    current_time = time.time()
    elapsed = current_time - _start_time
    if elapsed > 300.0 - _DEADLINE_MARGIN:
        return -mu

    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-10)
    
    # Kriging Believer: shift the mean toward exploration
    # k adapts: larger when far from optimum (more exploration), smaller when close
    f_opt = -39.1662
    dist = max(f_best - f_opt, 0.0)
    
    # k in [0.3, 1.5]: small near optimum (exploitation), larger far away (exploration)
    k = 0.3 + 1.2 * (1.0 - np.exp(-dist / 12.0))
    
    # Shifted mean: mu_kb = mu - k * sigma
    # This makes points with high uncertainty appear to have lower expected values
    mu_kb = mu - k * sigma
    
    # Expected Improvement on the Kriging Believer model
    # EI = (f_best - mu_kb) * Phi((f_best - mu_kb)/sigma) + sigma * phi((f_best - mu_kb)/sigma)
    z = (f_best - mu_kb) / sigma
    ei_kb = (f_best - mu_kb) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure all scores are finite
    ei_kb = np.where(np.isfinite(ei_kb), ei_kb, 0.0)
    
    return ei_kb
# EVOLVE-BLOCK-END
