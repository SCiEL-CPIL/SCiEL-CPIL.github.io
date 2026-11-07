"""Rewritten program: Cost-Aware Expected Improvement with Exploration."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Aware Expected Improvement (CA-EI) with exploration bonus.
    
    This acquisition function balances exploitation (Expected Improvement)
    and exploration (GP uncertainty) while accounting for evaluation costs.
    The score is defined as (EI + lambda * sigma) / cost, where lambda
    controls the trade-off between improvement and uncertainty.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best observed objective value so far.
    cost : ndarray, shape (2000,)
        Deterministic cost estimates for each candidate (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next).
    """
    # Clip sigma to avoid division by zero or negative values
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized distance from current best
    z = (f_best - mu) / sigma_safe
    
    # Standard Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: (EI + exploration_bonus) / cost
    # The exploration bonus encourages visiting high-uncertainty regions,
    # but is penalized by the cost of evaluation.
    # lambda = 0.1 is a conservative exploration weight.
    lambda_explore = 0.1
    
    # Calculate the combined acquisition value
    numerator = ei + lambda_explore * sigma_safe
    denominator = cost + 1e-9  # Avoid division by zero
    
    scores = numerator / denominator
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
