"""Rewritten acquisition for Bayesian optimization of 6d Hartmann."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement with an exploration bonus.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (n,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: encourages visiting high-uncertainty regions
    # The coefficient balances exploitation (EI) with exploration (sigma)
    exploration_weight = 0.1
    exploration_bonus = exploration_weight * sigma_safe
    
    # Combined score
    scores = ei + exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
