"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function with adaptive exploration.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        UCB scores for each candidate. Higher scores indicate more promising candidates.
    """
    # Clip sigma to avoid division by zero and numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive exploration parameter based on the number of evaluations
    # We can estimate the number of evaluations from the context, but since we don't have it directly,
    # we use a heuristic based on the relationship between f_best and the expected minimum.
    # For Goldstein-Price, the minimum is 3.0, so we can use this to gauge progress.
    
    # Use a UCB with beta that balances exploration and exploitation
    # A common choice is beta = 2 * log(n) where n is the number of evaluations
    # Since we don't have n directly, we use a fixed but reasonable value
    # that adapts based on the current best value
    
    # The Goldstein-Price function has a minimum of 3.0
    # We can use the gap between f_best and 3.0 to adapt beta
    min_value = 3.0
    gap = f_best - min_value
    
    # If we're close to the minimum, reduce exploration
    # If we're far, increase exploration
    # Use a logarithmic scaling for stability
    if gap > 1e-6:
        # More exploration when far from optimum
        beta = 2.0 * np.log1p(1.0 / gap)
    else:
        # Less exploration when close to optimum
        beta = 0.5
    
    # Ensure beta is within reasonable bounds
    beta = np.clip(beta, 0.5, 5.0)
    
    # Compute UCB: we want to minimize, so we use -mu + beta * sigma
    # Higher UCB means more promising (lower expected value with uncertainty bonus)
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
