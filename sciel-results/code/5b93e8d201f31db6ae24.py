import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Safety: ensure finite inputs and avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # Compute Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware adjustment using pure additive penalty
    # cost is in [1, 2], so cost - 1 is in [0, 1] (mean(x))
    # We use a deterministic penalty: EI - lambda * cost
    # This directly subtracts the cost, guiding search toward cheaper regions
    # while still allowing exploration of expensive areas if EI is high enough
    
    # Use a small lambda to avoid over-penalizing cost
    # Hartmann 3D values are in [-4, 0], so EI is typically O(0.1 - 1.0)
    # cost is in [1, 2], so lambda * cost is in [0.1, 0.2] for lambda=0.1
    lambda_cost = 0.08
    cost_penalty = lambda_cost * cost
    
    # Combine: higher EI and lower cost leads to higher score
    scores = ei - cost_penalty
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
