"""Acquisition function using Upper Confidence Bound (UCB) for minimization."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using UCB for minimization.
    
    For minimization, we want to find points with low expected value and/or high uncertainty.
    The Lower Confidence Bound (LCB) is defined as:
    LCB(x) = mu(x) - kappa * sigma(x)
    
    Since we need to return scores where *higher* is better (to select the next point),
    we maximize the negative LCB:
    Score(x) = -LCB(x) = -mu(x) + kappa * sigma(x)
    
    This balances exploitation (low mu) and exploration (high sigma).
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        The lowest observed value so far. (Not used in standard UCB, but kept for signature)
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Exploration-exploitation trade-off parameter
    # A value of 2.0 is a common heuristic for balancing exploration and exploitation
    # in low-dimensional Bayesian optimization.
    kappa = 2.0
    
    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 0.0)
    
    # Compute the score: -mu + kappa * sigma
    # This is equivalent to maximizing the negative Lower Confidence Bound
    scores = -mu + kappa * sigma_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
