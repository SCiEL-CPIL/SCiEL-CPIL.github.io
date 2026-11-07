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
    Thompson Sampling-style acquisition: draw a single sample from the GP posterior
    at each candidate point and score by that sample. This naturally balances
    exploration and exploitation without tuning a beta parameter, and avoids
    the scale-mismatch issues of EI/UCB hybrids.
    
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
    
    # Thompson Sampling: draw a sample from the posterior at each candidate
    # The sample is mu + sigma * z where z ~ N(0,1)
    # Higher sample = higher expected value, but we want to minimize, so
    # we negate: lower sample = better (more likely to be minimum)
    # Score = -sample, so higher score means lower sampled value (better for minimization)
    
    # Draw random standard normal samples
    rng = np.random.default_rng()
    z = rng.standard_normal(len(mu))
    
    # Sample from GP posterior
    samples = mu + sigma * z
    
    # For minimization, we want to select points with low samples
    # Score = -sample (higher score = lower sample = better)
    scores = -samples
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
