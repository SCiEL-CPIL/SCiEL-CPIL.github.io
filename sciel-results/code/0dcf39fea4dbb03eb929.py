"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: LCB with cost discounting and adaptive exploration.
    
    Uses the Lower Confidence Bound (LCB) criterion for minimization,
    which is structurally different from UCB/EI and handles multimodal
    functions well. The cost is applied as a discount to favor cheaper
    evaluations within the tight budget.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - lowest observed value
    cost : array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    --------
    score : array of shape (2000,) - acquisition scores (higher = evaluate next)
    """
    t0 = time.time()
    
    # Safety check for empty or invalid inputs
    if len(mu) == 0:
        return np.array([])
    
    # Ensure sigma is strictly positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute basic statistics for adaptive parameter tuning
    std_mu = np.std(mu)
    mean_sigma = np.mean(sigma_safe)
    
    # Adaptive beta: exploration-exploitation trade-off
    # When uncertainty is high relative to mean spread, explore more
    # When GP is confident, exploit more
    denom = std_mu + mean_sigma
    if denom < 1e-8:
        ratio = 1.0
    else:
        ratio = mean_sigma / denom
    
    # Map ratio [0, 1] to beta [1.0, 2.5]
    # Higher beta = more exploration (larger penalty on uncertainty)
    beta = 1.0 + ratio * 1.5
    
    # LCB for minimization: we want to minimize mu - beta * sigma
    # But since we return "higher is better" scores, we negate:
    # score = -(mu - beta * sigma) = -mu + beta * sigma
    lcb_score = -mu + beta * sigma_safe
    
    # Cost discounting: multiply by 1/cost^gamma
    # Costs are in [1, 2]. gamma=2.0 gives moderate preference for cheaper points
    # This is less aggressive than gamma=3.0, allowing some exploration of expensive regions
    gamma = 2.0
    cost_safe = np.maximum(cost, 1.0)
    cost_discount = 1.0 / (cost_safe ** gamma)
    
    # Final score: LCB discounted by cost
    score = lcb_score * cost_discount
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard
    elapsed = time.time() - t0
    if elapsed > 250:
        # Just return what we have; operations are fast
        return score
    
    return score
# EVOLVE-BLOCK-END
