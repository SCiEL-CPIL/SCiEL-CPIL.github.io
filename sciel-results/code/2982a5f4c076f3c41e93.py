"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Args:
        mu: (N,) array of GP mean predictions.
        sigma: (N,) array of GP standard deviations.
        f_best: float, best observed value so far.
        cost: (N,) array of costs for each candidate.
    
    Returns:
        (N,) array of acquisition scores (higher is better).
    """
    N = len(mu)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei_term = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei_term = np.maximum(ei_term, 0.0)
    
    # Cost-aware: divide by cost
    # Add small epsilon to cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-10)
    
    # Base score: EI per unit cost
    base_score = ei_term / cost_safe
    
    # Exploration bonus: encourage evaluating high-uncertainty, low-cost points
    # This helps the GP model converge faster by gathering information early
    exploration_bonus = 0.05 * sigma_safe / cost_safe
    
    # Combined score
    scores = base_score + exploration_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
