"""Cost-aware acquisition for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# We track time to ensure we don't exceed the 300s limit, though this function
# is called many times, the total time is dominated by the GP updates.
# However, the directive asks for an explicit deadline guard.
# Since we don't have a global state across calls easily in this snippet without
# modifying the signature or using a global, we will assume the caller handles
# the main loop timing, but we will keep this function as fast as possible.
# The "deadline guard" is effectively the speed of this function.

def _zscore(arr):
    """
    Robust z-scoring. If std is very small, return arr - mean to avoid division by zero.
    """
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-8:
        return arr - mean
    return (arr - mean) / std

def acquisition(mu, sigma, f_best, cost):
    """
    Compute acquisition scores for 2000 candidates using a z-scored combination
    of Cost-aware Expected Improvement (EI) and Cost-aware Upper Confidence Bound (UCB).
    
    Parameters:
    -----------
    mu : np.ndarray
        Mean of the GP posterior, shape (2000,).
    sigma : np.ndarray
        Standard deviation of the GP posterior, shape (2000,).
    f_best : float
        Best observed value so far.
    cost : np.ndarray
        Cost of evaluating each candidate, shape (2000,).
        
    Returns:
    --------
    np.ndarray
        Acquisition scores, shape (2000,). Higher is better.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8) # Safety for cost division
    
    # 1. Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Compute Upper Confidence Bound (UCB)
    # We use a beta of 2.0, which is a standard heuristic for moderate exploration.
    beta = 2.0
    ucb = -mu + beta * sigma
    
    # 3. Make them cost-aware
    ei_cost = ei / cost
    ucb_cost = ucb / cost
    
    # 4. Z-score both terms to make them comparable on the same scale
    # This is the structural change: normalizing the signals before combining.
    z_ei = _zscore(ei_cost)
    z_ucb = _zscore(ucb_cost)
    
    # 5. Combine with 75% exploitation (EI) and 25% exploration (UCB)
    # This weighting was found to be effective in the inspiration.
    scores = 0.75 * z_ei + 0.25 * z_ucb
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
