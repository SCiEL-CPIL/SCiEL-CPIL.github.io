"""Acquisition function for Bayesian optimization of goldstein price 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function that combines expected improvement (EI) with a small
    exploration bonus to avoid premature convergence.
    
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
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Guard against zero/negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardize for EI computation
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (minimization)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration bonus proportional to uncertainty
    # This helps escape local optima and explore more broadly
    exploration_bonus = 0.01 * sigma_safe
    
    scores = ei + exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
