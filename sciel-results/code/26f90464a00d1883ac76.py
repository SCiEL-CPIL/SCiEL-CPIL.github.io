"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 3d.
    
    Uses Lower Confidence Bound (LCB) to balance exploration and exploitation.
    LCB encourages exploration by adding a bonus proportional to the uncertainty
    (standard deviation). This is particularly effective for functions with
    multiple local minima where pure exploitation (like EI) can get stuck.
    
    The score is defined as -LCB(x) so that higher scores correspond to lower
    confidence bounds (more promising points for minimization).
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition values; higher means more promising
    """
    # Ensure sigma is non-negative to avoid negative exploration bonuses
    sigma = np.maximum(sigma, 0.0)
    
    # Exploration-exploitation trade-off parameter
    # A value of 1.0 is a standard starting point for LCB.
    # Higher values encourage more exploration.
    beta = 1.0
    
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to maximize the score, so we return -LCB(x)
    # score = -mu(x) + beta * sigma(x)
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
