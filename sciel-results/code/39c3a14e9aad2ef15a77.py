"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: we must finish well within 330s wall-clock total.
# We track elapsed time to ensure we don't exceed budget.
_DEADLINE = 320.0  # safety margin
_START_TIME = time.time()


def _check_deadline():
    """Check if we're approaching the deadline."""
    return (time.time() - _START_TIME) > _DEADLINE


def acquisition(mu, sigma, f_best):
    """
    Quantile-based acquisition: scores candidates by posterior 5th percentile.
    
    score = mu - 1.645 * sigma
    
    This is equivalent to LCB (Lower Confidence Bound) with beta=1.645,
    which corresponds to the 5th percentile of the Gaussian posterior.
    Lower values of (mu - 1.645*sigma) indicate more promising candidates
    for minimization, so we negate to make higher scores better.
    
    Parameters:
        mu: array of shape (2000,) - posterior means in objective units
        sigma: array of shape (2000,) - posterior standard deviations
        f_best: float - lowest observed value so far
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    # Ensure we're not over budget
    if _check_deadline():
        # Return a safe fallback: just use -mu (exploit best mean)
        return -mu.copy()
    
    # Quantile-based lower bound: 5th percentile estimate
    # For a Gaussian, the 5th percentile is mu - 1.645 * sigma
    # We want to MINIMIZE the objective, so lower quantile estimates are better.
    # Acquisition score should be HIGHER for more promising candidates.
    # So we negate the lower bound: score = -(mu - 1.645 * sigma) = -mu + 1.645 * sigma
    
    # Clip sigma to avoid numerical issues
    sigma_clipped = np.maximum(sigma, 1e-12)
    
    # The 5th percentile (lower quantile) of the posterior
    lower_quantile = mu - 1.645 * sigma_clipped
    
    # Higher score = more promising for minimization
    # We want candidates with LOW lower_quantile to get HIGH scores
    score = -lower_quantile
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
