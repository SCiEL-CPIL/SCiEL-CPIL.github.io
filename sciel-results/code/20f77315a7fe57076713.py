"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of Expected Improvement (EI) and a penalty term
    that discourages expensive evaluations, with an exploration bonus
    for high-uncertainty regions.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions at candidate points.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: clip sigma to avoid division by zero
    sigma = np.clip(sigma, 1e-9, None)
    
    # Clip cost to avoid division by zero
    cost = np.clip(cost, 1e-9, None)
    
    # Compute Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scaling: divide EI by cost to get expected improvement per unit cost
    ei_per_cost = ei / cost
    
    # Add a small exploration bonus for high-uncertainty regions
    # This helps escape local optima in the rastrigin function
    exploration_bonus = 0.1 * sigma / cost
    
    # Combine: primary signal is EI per cost, with a small exploration term
    scores = ei_per_cost + exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
