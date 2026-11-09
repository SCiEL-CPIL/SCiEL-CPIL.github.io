"""Rewritten acquisition for Bayesian optimization of Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Expected Improvement (EI) with a small exploration
    component to balance exploitation and exploration.

    Parameters
    ----------
    mu : ndarray, shape (n,)
        GP mean predictions at candidate points.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.

    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher means more promising.
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration bonus proportional to uncertainty
    # This helps escape local plateaus in the search
    exploration_weight = 0.1
    scores = ei + exploration_weight * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
