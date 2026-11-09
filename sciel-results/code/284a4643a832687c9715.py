"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for time budget tracking
_start_time = None


def acquisition(mu, sigma, f_best):
    """
    Quantile-based acquisition for 1D sphere minimization.
    
    Structural change: Uses the predicted 5th percentile (mu - 1.645 * sigma)
    as the acquisition score. This directly targets the lower tail of the
    posterior, providing aggressive exploitation of low predicted values
    while still accounting for uncertainty in a principled way.
    
    For the unimodal sphere function x^2 on [-5,5], this approach rapidly
    converges to the global minimum at x=0 by prioritizing candidates with
    the lowest predicted quantile.
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
    
    # Quantile-based acquisition: maximize the predicted 5th percentile
    # For minimization, we want the lowest predicted value, so we use
    # the lower quantile: mu - z * sigma where z = 1.645 (5th percentile)
    # Higher score = better candidate to evaluate next
    
    # The 5th percentile is mu - 1.645 * sigma
    # We want to maximize this (higher = more promising for minimization)
    # But wait: for minimization, we want LOW values. So we should
    # minimize (mu - 1.645 * sigma), which means we want to maximize
    # -(mu - 1.645 * sigma) = -mu + 1.645 * sigma
    
    # Actually, let me reconsider. The quantile approach for minimization:
    # We want to find the candidate with the lowest possible value.
    # The 5th percentile of the posterior is mu - 1.645 * sigma.
    # We want to MINIMIZE this quantity (find the lowest possible value).
    # So the acquisition score should be the NEGATIVE of this:
    # score = -(mu - 1.645 * sigma) = -mu + 1.645 * sigma
    
    # This gives higher scores to candidates with low mu AND high sigma,
    # which is a balanced exploration-exploitation strategy.
    
    scores = -mu + 1.645 * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
