"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're getting close to the 300s limit, return safe scores
    # We can't know the exact deadline here, but we can add a safety margin
    # The cost parameter gives us a hint about progress
    # cost = 1 + mean(x_scaled) where x_scaled is in [0,1]
    # So cost is in [1, 2]
    
    # Use a simpler, faster acquisition that's robust
    # Expected Improvement with cost normalization
    # Add a small penalty for high cost to prefer cheaper evaluations early
    
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware: divide by cost, but also add a small exploration bonus
    # for points with low sigma (more certain) to avoid over-exploration
    cost_safe = np.maximum(cost, 1e-9)
    
    # Use EI / cost^alpha with alpha < 1 to be less aggressive on cost
    score = ei / (cost_safe ** 0.5)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
