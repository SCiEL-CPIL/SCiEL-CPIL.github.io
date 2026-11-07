"""Acquisition function for Bayesian optimization of Hartmann 6D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function that combines Expected Improvement with an exploration bonus.
    
    Uses a hybrid approach:
    - Base: Expected Improvement (EI) for exploitation
    - Bonus: UCB-like term for exploration in uncertain regions
    - Handles numerical stability
    
    Parameters:
    -----------
    mu : ndarray, shape (n_candidates,)
        Posterior mean at candidate points
    sigma : ndarray, shape (n_candidates,)
        Posterior standard deviation at candidate points
    f_best : float
        Best observed value so far
    
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        Acquisition scores (higher = more promising)
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Numerical stability: clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: UCB-style term that rewards high uncertainty
    # Use a moderate beta to balance exploration/exploitation
    beta = 0.5
    ucb_bonus = beta * sigma_safe
    
    # Combine: EI + exploration bonus
    # The EI already has some exploration properties, but adding a small UCB term
    # helps escape local optima in high-dimensional spaces
    scores = ei + ucb_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
