"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline management
_START_TIME = time.time()
_DEADLINE_SECONDS = 285.0  # Safety margin before 300s limit

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
    # Improvement is defined as f_best - f(x)
    # We want to maximize E[improvement]
    
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -50, 50)
    
    # EI formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # EI should be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB (Upper Confidence Bound) for minimization
    # UCB = mu - kappa * sigma
    # Lower UCB means more promising
    kappa = 2.0
    ucb = mu - kappa * sigma
    
    # Normalize EI and UCB to [0, 1] for fair comparison
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    if ucb_max - ucb_min > 1e-12:
        ucb_norm = 1.0 - (ucb - ucb_min) / (ucb_max - ucb_min)  # Invert: lower UCB = higher norm
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Combine EI and UCB with a weighted sum
    # Weight EI more heavily, but still consider exploration via UCB
    weight_ei = 0.7
    weight_ucb = 0.3
    
    combined = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Cost-aware scoring:
    # We want to maximize improvement per unit cost.
    # Use a non-linear cost penalty: divide by (cost - offset) to amplify low-cost advantage
    # Offset 0.6 increases relative weight of low-cost candidates
    cost_offset = 0.6
    cost_denom = cost - cost_offset
    cost_denom = np.maximum(cost_denom, 0.1)  # Avoid division by zero or negative
    
    score = combined / cost_denom
    
    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
# EVOLVE-BLOCK-END
