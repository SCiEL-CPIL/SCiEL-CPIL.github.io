"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Posterior Mean Minimization (PMM) acquisition function.
    
    Directly exploits the GP predictive mean by selecting the candidate
    with the lowest expected objective value. This ignores the predictive
    uncertainty (sigma), forcing a deterministic exploitation strategy.
    
    Args:
        mu: Array of predicted means, shape (2000,).
        sigma: Array of predicted standard deviations, shape (2000,).
        f_best: The best observed objective value so far.
        
    Returns:
        Array of scores, shape (2000,). Higher scores are preferred.
    """
    # We want to minimize the objective function.
    # The point with the lowest mean (mu) is the best candidate.
    # To convert "lowest is best" to "highest score is best", we negate mu.
    scores = -mu
    
    # Ensure all scores are finite.
    # If any mu is NaN or Inf, replace with a very low score (negative infinity)
    # so that it is never selected.
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
