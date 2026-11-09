"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    UCB (Upper Confidence Bound) acquisition function for minimization.
    
    Parameters:
    -----------
    mu : np.ndarray
        Mean of the GP posterior, shape (2000).
    sigma : np.ndarray
        Standard deviation of the GP posterior, shape (2000).
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : np.ndarray
        UCB scores, shape (2000). Higher scores indicate more promising points.
    """
    # Clip sigma to avoid numerical instability
    epsilon = 1e-8
    sigma_safe = np.maximum(sigma, epsilon)
    
    # UCB for minimization: we want to find points likely to have low values.
    # We use a lower confidence bound (LCB) which is mu - beta * sigma.
    # Higher LCB scores (less negative or more positive) indicate more promising points.
    # Alternatively, we can maximize -LCB = -mu + beta * sigma.
    
    # Use a beta value that balances exploration and exploitation.
    # A common choice is beta = 2.0, but we can adjust based on the scale.
    # Since we're minimizing, we want to explore regions with high uncertainty (high sigma)
    # and regions with low predicted values (low mu).
    
    # Let's use a dynamic beta based on the number of evaluations.
    # However, since we don't have access to the step count directly, we'll use a fixed beta.
    # A value of 2.0 is a reasonable starting point for UCB.
    beta = 2.0
    
    # LCB = mu - beta * sigma (for minimization)
    # We want to maximize this, so we return -LCB = -mu + beta * sigma
    scores = -mu + beta * sigma_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
