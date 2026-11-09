"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Regret (ER) normalized by cost.
    
    Expected Regret is defined as E[max(0, f(x) - f_best)] where f(x) ~ N(mu, sigma^2).
    This is mathematically equivalent to Expected Improvement but framed as regret minimization.
    We normalize by cost to favor evaluations that reduce expected regret per unit cost.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed objective value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher is better).
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Regret: E[max(0, f(x) - f_best)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    er = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure non-negative (numerical stability)
    er = np.maximum(er, 0.0)
    
    # Normalize by cost to get expected regret reduction per unit cost
    # Ensure cost is positive to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = er / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
