"""Rewritten program using a different acquisition strategy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function for cost-aware Bayesian optimization.
    
    Uses a variant of Expected Improvement (EI) with cost-aware weighting.
    The key insight is to use a more robust formulation that handles
    sigma near zero better and weights by inverse cost more aggressively.
    
    We use the 'probability of improvement' weighted EI, which can be more
    stable, combined with a cost-normalized term.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean predictions of the GP
    sigma : ndarray, shape (2000,)
        Standard deviation predictions of the GP
    f_best : float
        Best observed value so far (minimum)
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z = np.clip(z, -50, 50)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0)
    
    # Cost-aware: divide by cost, but use a softened cost to avoid division by near-zero
    # The cost is 1 + mean(x) where x in [0,1], so cost is in [1, 2]
    # We want to strongly prefer cheaper evaluations
    # Use cost^2 to make the preference for cheap evaluations stronger
    cost_safe = np.maximum(cost, 1e-6)
    
    # Primary term: EI per unit cost, with stronger cost weighting
    score = ei / (cost_safe ** 2)
    
    # Add a small exploration bonus for high uncertainty relative to cost
    # This helps when EI is low but sigma is high (exploration)
    exploration_bonus = sigma_safe / (cost_safe ** 1.5)
    
    # Combine: weight EI term heavily, exploration lightly
    score = score + 0.1 * exploration_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
