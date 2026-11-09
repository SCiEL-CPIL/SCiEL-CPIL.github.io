# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    Uses a hybrid strategy: Expected Improvement with an exploration bonus
    that adapts based on the current uncertainty landscape.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        The best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : array-like, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: encourage visiting high-uncertainty regions
    # Scale the exploration bonus by the relative uncertainty
    # Use a moderate exploration weight to balance exploitation and exploration
    exploration_weight = 0.15
    exploration_bonus = exploration_weight * sigma_safe
    
    # Combined score: EI + exploration bonus
    # This encourages both finding improvements and exploring uncertain regions
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
