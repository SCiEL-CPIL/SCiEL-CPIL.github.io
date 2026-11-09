"""Rewritten acquisition for cost-aware BO on Griewank-2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Probability of Improvement (PI) divided by cost.
    
    PI is bounded and sigmoidal, which avoids the unbounded exploration of EI
    in high-uncertainty regions. Dividing by cost accounts for the fact that
    evaluating points with higher coordinates is more expensive.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Probability of Improvement: P(f(x) < f_best)
    # = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Cost-aware score: PI / cost
    # Use a small epsilon to avoid division by zero
    cost_safe = np.maximum(cost, 1e-12)
    scores = pi / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
