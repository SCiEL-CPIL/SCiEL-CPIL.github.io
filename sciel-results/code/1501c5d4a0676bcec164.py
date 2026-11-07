"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a combination of Expected Improvement (EI) and a penalty for high cost,
    but with a more robust formulation that handles edge cases better.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at each candidate point.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at each candidate point.
    f_best : float
        Best (lowest) observed function value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to get improvement per unit cost
    # But also add a small exploration bonus for high uncertainty
    # to avoid getting stuck in local optima
    
    # Base score: EI per unit cost
    base_score = ei / cost
    
    # Exploration bonus: reward high uncertainty (sigma) but penalize if cost is high
    # This encourages exploring uncertain regions while being mindful of cost
    exploration_bonus = 0.1 * sigma_safe / cost
    
    # Combine base score and exploration bonus
    scores = base_score + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
