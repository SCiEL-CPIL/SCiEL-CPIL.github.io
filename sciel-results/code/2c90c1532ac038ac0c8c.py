"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit

    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)

    # Normalize cost to [0, 1] range for adaptive beta
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range < 1e-10:
        cost_norm = np.zeros_like(cost)
    else:
        cost_norm = (cost - cost_min) / cost_range

    # Adaptive beta: high exploration early, low exploitation late
    # Use a smooth decay based on cost position
    # beta starts high (exploration) and decreases to near 0 (exploitation)
    # We use a power-law decay: beta = beta_max * (1 - cost_norm)^power
    # This gives aggressive exploration for low-cost points and
    # strict exploitation for high-cost points
    
    beta_max = 3.0  # Maximum exploration bonus
    power = 1.5  # Decay rate
    
    # Cost-aware UCB: mu - beta * sigma, but we want to favor low cost
    # Score = -mu + beta * sigma * cost_factor
    # where cost_factor decreases with cost
    cost_factor = (1.0 - cost_norm) ** power
    beta = beta_max * cost_factor
    
    # UCB score (higher is better for minimization: we want low mu, high uncertainty)
    # For minimization: score = -mu + beta * sigma
    # But we also want to penalize high cost
    scores = -mu + beta * sigma_safe
    
    # Apply additional cost penalty: divide by a cost-aware factor
    # Lower cost should be preferred
    # Use: score = (-mu + beta * sigma) / (1 + cost_norm)
    # This makes low-cost points more attractive
    
    scores = scores / (1.0 + cost_norm)
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Safety margin check
    if time.time() > deadline:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
