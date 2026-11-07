import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a blend of Expected Improvement and LCB,
    with variance penalty for stability in high dimensions.
    
    Args:
        mu: (2000,) array of GP means.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, the best observed value so far.
        cost: (2000,) array of evaluation costs.
        
    Returns:
        (2000,) array of scores, higher is better.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Lower Confidence Bound (LCB) for minimization:
    # We want to minimize, so lower values are better.
    # LCB = mu - beta * sigma, and we maximize -LCB = -mu + beta * sigma
    # Choose beta to balance exploration and exploitation
    beta = 2.0
    
    # LCB score (higher is better for minimization)
    lcb_score = -mu + beta * sigma_safe
    
    # Combine EI and LCB
    # Blend them with weights
    blend = 0.6 * ei + 0.4 * lcb_score
    
    # Add a variance penalty to stabilize GP predictions in high dimensions
    # This reduces sensitivity to noise and improves consistent exploration
    # The penalty term encourages exploration of uncertain regions
    variance_penalty = 0.1 * sigma_safe
    
    # Final score: blend + variance penalty, normalized by cost
    score = (blend + variance_penalty) / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
