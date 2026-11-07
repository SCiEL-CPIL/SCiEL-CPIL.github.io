"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses Expected Improvement (EI) normalized by cost, with a small
    exploration bonus based on uncertainty per unit cost.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of GP predictions
    sigma : array of shape (2000,)
        Standard deviation of GP predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = more promising)
    """
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # Standard EI (expected improvement in objective value, lower is better)
    delta = f_best - mu
    z = delta / sigma_safe
    ei = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware EI: improvement per unit cost
    cea = ei / cost_safe
    
    # Small exploration term: uncertainty per unit cost
    exploration = sigma_safe / cost_safe
    
    # Combine: primarily exploit EI/cost, with small exploration bonus
    # The weight on exploration is small to avoid over-exploring
    scores = cea + 0.05 * exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
