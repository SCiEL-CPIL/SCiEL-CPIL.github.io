"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline management
_START_TIME = time.time()
_DEADLINE_SECONDS = 290.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        return np.zeros_like(mu, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle any NaN or Inf values
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    # Ensure sigma is strictly positive for division
    sigma = np.maximum(sigma, 1e-12)
    
    # Ensure cost is within reasonable bounds for Rastrigin task
    # Cost is 1 + mean(x_scaled), so [1, 2]
    cost = np.clip(cost, 1.0, 2.0)
    
    # Compute Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Structural Change: Pure EI with aggressive cost penalty.
    # Instead of blending EI and UCB, we rely on EI's natural exploration/exploitation balance.
    # The cost penalty is modified to (cost - 0.5) to amplify the difference between 
    # low-cost (1.0 -> 0.5) and high-cost (2.0 -> 1.5) regions.
    # This effectively triples the relative weight of the cost difference compared to 
    # a simple linear cost penalty, steering the search more aggressively towards 
    # the low-cost region (origin in scaled coordinates) where the GP is likely 
    # more accurate and the function is easier to navigate.
    
    cost_penalty = cost - 0.5
    # Ensure no division by zero (cost >= 1.0, so cost_penalty >= 0.5)
    cost_penalty = np.maximum(cost_penalty, 1e-6)
    
    score = ei / cost_penalty
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
