"""Rewritten program using a different acquisition approach (UCB with cost-aware scaling)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function.
    
    Uses Upper Confidence Bound (UCB) with a cost-aware scaling factor.
    The UCB balances exploration (via sigma) and exploitation (via mu),
    while the cost-aware term penalizes expensive evaluations.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure no division by zero or negative values
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Standard UCB: -mu + beta * sigma
    # We want to minimize, so higher score = more promising
    # UCB for minimization: score = -mu + beta * sigma
    
    # Adaptive beta based on the number of evaluations implied by cost
    # Use a simple heuristic: beta increases slightly with exploration needs
    # Normalize cost to get a sense of relative expense
    cost_mean = np.mean(cost_safe)
    
    # Beta parameter: balance exploration/exploitation
    # Start with a moderate beta and scale by cost considerations
    beta = 2.0
    
    # Base UCB score for minimization
    ucb_score = -mu + beta * sigma_safe
    
    # Cost-aware scaling: penalize high-cost points
    # We want to prefer low-cost points, so divide by cost or use a penalty
    # Using division by cost encourages cheaper evaluations
    cost_aware_score = ucb_score / cost_safe
    
    # Ensure finite values
    cost_aware_score = np.where(np.isfinite(cost_aware_score), cost_aware_score, 0.0)
    
    return cost_aware_score
# EVOLVE-BLOCK-END
