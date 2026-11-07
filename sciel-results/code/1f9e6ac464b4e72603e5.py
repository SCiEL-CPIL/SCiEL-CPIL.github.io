"""Cost-aware acquisition for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Precompute constants for speed
# The cost model is 1 + mean(x scaled to [0,1]) where x is in [-1, 1]^8.
# Since x is uniform in [-1, 1], the scaled x is uniform in [0, 1].
# However, the acquisition function receives `cost` directly.
# We need to handle the case where cost might be zero or very small, though
# the problem states cost is 1 + mean(...), so it's always >= 1.

def acquisition(mu, sigma, f_best, cost):
    """
    Compute acquisition scores for 2000 candidates.
    
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
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute Upper Confidence Bound (UCB)
    # UCB = -mu + beta * sigma
    # We need to choose beta. A common choice is based on the log of the number of samples,
    # but since we don't have that directly, we can use a fixed beta or derive it from cost.
    # For cost-awareness, we divide by cost.
    # Let's use a beta that balances exploration and exploitation.
    # Based on the inspiration, a z-scored combination of EI/cost and UCB/cost worked well.
    # Let's try a simpler approach first: just EI / cost.
    # But the inspiration says "Z-scored combination of EI/cost and UCB/cost with 25% exploration weight".
    
    # Let's implement the z-scored combination.
    # First, compute the cost-adjusted terms.
    ei_cost = ei / cost
    ucb = -mu + 2.0 * sigma  # beta = 2.0 is a reasonable default
    ucb_cost = ucb / cost
    
    # Z-score both terms to make them comparable
    def zscore(arr):
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-8:
            return arr - mean
        return (arr - mean) / std
    
    z_ei = zscore(ei_cost)
    z_ucb = zscore(ucb_cost)
    
    # Combine with 25% exploration weight (i.e., 25% UCB, 75% EI)
    scores = 0.75 * z_ei + 0.25 * z_ucb
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
