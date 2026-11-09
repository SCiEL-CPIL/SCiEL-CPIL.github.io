"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Lower Confidence Bound (LCB) acquisition function for minimization.
    
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
        LCB scores, shape (2000). Higher scores indicate more promising points.
    """
    # LCB for minimization: minimize mu - beta * sigma
    # To maximize the score, we use: score = -mu + beta * sigma
    # beta controls the trade-off between exploitation (mu) and exploration (sigma)
    # A value of 2.0 is a common starting point for small budgets
    beta = 2.0
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 0.0)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 0.0)
    
    # Calculate scores
    scores = -mu + beta * sigma_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
