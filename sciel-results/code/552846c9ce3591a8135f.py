"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for time budget tracking
_start_time = None


def acquisition(mu, sigma, f_best):
    """
    Quantile-based acquisition for 1D sphere minimization.
    
    Structural change: Use the predicted 5th percentile of the posterior,
    mu - 1.645 * sigma, as the acquisition score. This directly targets
    the lower tail of the predictive distribution, providing an aggressive
    exploitation signal that is more robust than UCB's additive bonus.
    
    For minimization, we want to maximize the acquisition score, so we
    negate the quantile: - (mu - 1.645 * sigma) = -mu + 1.645 * sigma.
    
    The coefficient 1.645 corresponds to the 5th percentile of the
    standard normal distribution, making this a principled choice
    for targeting the lower tail.
    """
    global _start_time
    
    # Time budget guard
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    # If we're close to the 330s limit, return a simple fallback
    if elapsed > 320:
        # Fallback: just use -mu (exploitation only)
        scores = -mu
        return np.where(np.isfinite(scores), scores, 0.0)
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Quantile-based acquisition: maximize the 5th percentile
    # The 5th percentile is mu - 1.645 * sigma
    # For minimization, we want to maximize - (5th percentile) = -mu + 1.645 * sigma
    quantile_score = -mu + 1.645 * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(quantile_score), quantile_score, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
