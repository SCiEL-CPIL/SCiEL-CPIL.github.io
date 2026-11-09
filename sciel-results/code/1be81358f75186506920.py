"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a hybrid of Expected Improvement (EI) and UCB-style exploration,
    weighted by a cost-awareness factor. The key insight is that we should
    balance exploitation (high EI) with exploration (high uncertainty) while
    accounting for the cost of evaluation.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate (1 + mean(x scaled to [0,1]))
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound component
    beta = 0.5  # Exploration parameter
    ucb = mu + beta * sigma_safe
    
    # Cost-awareness: penalize high-cost evaluations
    # Normalize costs to [0, 1] range for scaling
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min + 1e-9
    cost_normalized = (cost - cost_min) / cost_range
    
    # Combine EI and UCB with cost awareness
    # Higher EI and lower cost should lead to higher scores
    # We use a multiplicative cost penalty
    cost_penalty = 1.0 / (1.0 + cost_normalized)
    
    # Weighted combination: primarily EI, with some UCB exploration
    # The weight on UCB decreases as we get more confident (smaller sigma)
    confidence = 1.0 / (1.0 + sigma_safe)
    ucb_weight = 0.3 * (1.0 - confidence) + 0.1  # Between 0.1 and 0.4
    
    # Final score: combine EI and UCB, then apply cost penalty
    scores = (ei + ucb_weight * ucb) * cost_penalty
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
