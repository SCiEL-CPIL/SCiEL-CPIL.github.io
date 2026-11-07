import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function that combines expected improvement
    with a cost-aware exploration bonus.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviations
        f_best: float - best observed value so far
        cost: array of shape (2000,) - evaluation costs
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = better)
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: divide by cost to get per-unit-cost improvement
    cost_safe = np.maximum(cost, 1e-10)
    cost_aware_ei = ei / cost_safe
    
    # Cost-aware exploration bonus
    # The baseline used a non-cost-aware exploration term.
    # By dividing by cost, we ensure that we prefer cheaper regions when uncertain.
    exploration_bonus = sigma_safe / (sigma_safe + 1.0)
    cost_aware_exploration = exploration_bonus / cost_safe
    
    # Combine: primary signal is cost-aware EI, secondary is cost-aware exploration
    # We increase the exploration weight slightly to compensate for the cost penalty
    score = cost_aware_ei + 0.5 * cost_aware_exploration
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
