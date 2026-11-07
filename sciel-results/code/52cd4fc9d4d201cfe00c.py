"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) normalized by cost.
    Includes a small exploration bonus to prevent premature convergence.
    """
    # Deadline guard
    t0 = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Numerical stability: EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Exploration bonus
    # Using a small coefficient to encourage exploration without overwhelming exploitation
    exploration_weight = 0.05
    bonus = exploration_weight * sigma_safe
    
    # Numerator: EI + Bonus
    numerator = ei + bonus
    
    # Denominator: Cost
    # Cost is 1 + mean(x_scaled), so it is in [1, 2]
    cost_safe = np.maximum(cost, 1e-9)
    
    # Score: Higher is better
    score = numerator / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline (should be well within)
    if time.time() - t0 > 250:
        return score
        
    return score
# EVOLVE-BLOCK-END
