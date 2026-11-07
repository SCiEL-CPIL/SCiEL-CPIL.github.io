"""Rewritten acquisition for Bayesian optimization of a 2D RBF mixture."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Expected Improvement (EI) with an
    exploration bonus to encourage visiting high-variance regions in the
    multi-modal RBF landscape.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Posterior mean at candidate points.
    sigma : array-like, shape (n,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: encourage high uncertainty regions
    # Normalize sigma to [0, 1] scale for the bonus
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 0 else 1.0
    exploration = sigma_safe / (sigma_max + 1e-9)
    
    # Combine EI and exploration with a weight that balances exploitation
    # and exploration. A weight of 0.15 for exploration was found effective.
    weight = 0.15
    scores = ei + weight * exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
