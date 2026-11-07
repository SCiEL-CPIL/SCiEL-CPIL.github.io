"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're getting close to the 330s limit, return safe scores
    if time.time() > _DEADLINE:
        # Return simple scores that favor low cost and high uncertainty
        return (1.0 / np.maximum(cost, 1e-9)) * np.maximum(sigma, 1e-9)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Structural change: Use a probability-based acquisition function
    # Maximize P(f(x) < f_best - epsilon) / cost
    # This focuses on high-confidence improvements over a fixed threshold
    
    # Choose epsilon as a fraction of the spread of mu values
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min
    if mu_range < 1e-10:
        epsilon = 0.01 * max(abs(f_best), 1.0)
    else:
        epsilon = 0.05 * mu_range
    
    # For each point, compute P(f(x) < f_best - epsilon | mu, sigma)
    # Under normal distribution: P(f < t) = Phi((t - mu) / sigma)
    target = f_best - epsilon
    z_prob = (target - mu) / sigma_safe
    z_prob = np.clip(z_prob, -10, 10)
    
    # Probability of improvement beyond threshold
    prob_improve = norm.cdf(z_prob)
    
    # Normalize by cost to prefer cheaper evaluations
    score = prob_improve / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
