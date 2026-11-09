"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        Mean predictions from GP.
    sigma : array_like, shape (2000,)
        Standard deviation predictions from GP.
    f_best : float
        Best (lowest) observed function value.
    cost : array_like, shape (2000,)
        Evaluation cost for each candidate (1 + mean(x_scaled_to_01)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition values; higher is better.
    """
    # Ensure arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: handle any non-finite values in inputs
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    cost = np.maximum(cost, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Add exploration term weighted by sigma to encourage exploration
    # This helps escape local optima and find better minima
    exploration = 0.1 * sigma
    
    # Combine exploitation (EI) and exploration
    combined = ei + exploration
    
    # Cost-aware acquisition: combined score per unit cost
    scores = combined / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
