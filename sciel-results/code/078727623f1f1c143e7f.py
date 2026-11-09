"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Griewank 2D.
    
    Uses Expected Improvement (EI) with an exploration bonus, normalized by cost.
    The exploration bonus (xi * sigma) helps escape local minima in the multimodal
    Griewank function. The cost normalization accounts for the fact that
    evaluations near the center of the domain (lower scaled coordinates) are cheaper.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the Gaussian process at 2000 candidate points.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP at 2000 candidate points.
    f_best : float
        Best (lowest) observed objective value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point (1 + mean of scaled coordinates).
    
    Returns:
    --------
    scores : numpy.ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Safety: ensure we don't run too long
    # Compute EI-based acquisition with exploration term
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardize: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (minimization)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Exploration bonus: encourages visiting high-uncertainty regions
    # This is especially useful for multimodal functions like Griewank
    xi = 0.5  # Exploration exploitation trade-off
    exploration_bonus = xi * sigma_safe
    
    # Combined acquisition value
    combined = ei + exploration_bonus
    
    # Normalize by cost (higher cost = lower priority)
    # Ensure cost is at least 1 to avoid division issues
    cost_safe = np.maximum(cost, 1.0)
    
    scores = combined / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
