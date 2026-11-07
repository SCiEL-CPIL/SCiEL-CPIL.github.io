"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Expected Improvement with adaptive cost discounting.
    
    Uses Expected Improvement (EI) for minimization, discounted by evaluation cost.
    The discounting uses a smooth, adaptive power law that balances exploration
    (high uncertainty) with exploitation (high improvement) while penalizing
    expensive evaluations.
    
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
    
    n = len(mu)
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute the improvement: how much better than f_best could we get?
    # For minimization, improvement = f_best - mu
    improvement = f_best - mu
    
    # Standardized improvement
    z = improvement / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -20, 20)
    
    # Compute Expected Improvement for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    #       = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    ei = improvement * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware discounting with adaptive power
    # The cost ranges from 1.0 to 2.0 (since cost = 1 + mean(x_scaled))
    # We want to strongly penalize expensive evaluations but not completely ignore
    # the potential improvement
    
    cost_safe = np.maximum(cost, 1.0)
    
    # Adaptive gamma: use a moderate discount that balances exploration and exploitation
    # A lower gamma (closer to 1) means less penalty for cost, higher gamma means more
    # Based on the cost structure (1 to 2), gamma=1.5 provides a good balance
    gamma = 1.5
    
    # Cost discount factor
    cost_discount = cost_safe ** gamma
    
    # Final score: EI discounted by cost
    score = ei / cost_discount
    
    # Add a small exploration bonus for high-uncertainty regions
    # This helps escape local optima by giving some credit to uncertain areas
    # The bonus is scaled by the relative uncertainty and is small enough
    # not to dominate the EI term
    uncertainty_bonus = 0.1 * sigma_safe / cost_discount
    
    # Combine EI and exploration bonus
    score = score + uncertainty_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Ensure non-negative scores
    score = np.maximum(score, 0.0)
    
    # Deadline guard
    elapsed = time.time() - t0
    if elapsed > 290:
        pass
    
    return score
    # EVOLVE-BLOCK-END
