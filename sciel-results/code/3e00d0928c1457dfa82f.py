"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization using a hybrid approach
    combining Expected Improvement with adaptive exploration.
    
    Uses a Knowledge Gradient-inspired weighting to balance exploitation
    and exploration based on the current state of the search.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Ensure sigma is non-negative and has a small floor to avoid division issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Adaptive exploration component
    # Use the ratio of uncertainty to the range of means to scale exploration
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-10:
        mu_range = 1e-10
    
    # Scale exploration by the relative uncertainty
    # When uncertainty is high relative to the mean spread, explore more
    exploration_scale = np.clip(sigma / (mu_range + 1e-10), 0.0, 1.0)
    
    # Base exploration weight that adapts to the search state
    # Early in the search (high f_best relative to mu range), explore more
    # Late in the search, exploit more
    base_exploration_weight = 0.5
    
    # Compute exploration score using a square root to balance the scale
    exploration = base_exploration_weight * np.sqrt(sigma) * (1.0 + exploration_scale)
    
    # Combine EI and exploration with adaptive weighting
    # Use a softmax-like blending based on the relative magnitude
    ei_scale = np.max(ei) if np.max(ei) > 1e-10 else 1e-10
    exploration_scale_max = np.max(exploration) if np.max(exploration) > 1e-10 else 1e-10
    
    # Normalize both components to have similar scales
    ei_normalized = ei / ei_scale
    exploration_normalized = exploration / exploration_scale_max
    
    # Blend with a weight that favors exploitation as the search progresses
    # Use the ratio of f_best to the minimum mu as a proxy for progress
    min_mu = np.min(mu)
    if f_best > min_mu + 1e-10:
        progress = 1.0 - (f_best - min_mu) / (f_best - min_mu + 1e-10)
    else:
        progress = 1.0
    
    # Clip progress to [0, 1]
    progress = np.clip(progress, 0.0, 1.0)
    
    # Weight: more exploration early, more exploitation later
    exploration_weight = 0.3 + 0.4 * (1.0 - progress)
    exploitation_weight = 1.0 - exploration_weight
    
    scores = exploitation_weight * ei_normalized + exploration_weight * exploration_normalized
    
    # Scale back to original EI scale for meaningful comparison
    scores = scores * ei_scale
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
