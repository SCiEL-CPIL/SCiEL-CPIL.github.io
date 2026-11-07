"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of goldstein price 2d.
    
    Parameters:
    - mu: array of shape (2000,) - posterior mean at candidate points
    - sigma: array of shape (2000,) - posterior std at candidate points
    - f_best: float - best observed value so far
    
    Returns:
    - scores: array of shape (2000,) - acquisition scores, higher means evaluate next
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration term based on uncertainty to avoid getting stuck
    # This encourages exploring high-uncertainty regions
    # But keep it small so EI dominates
    exploration_weight = 0.1
    scores = ei + exploration_weight * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
