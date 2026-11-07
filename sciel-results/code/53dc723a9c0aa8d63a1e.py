"""Cost-aware acquisition function for Bayesian optimization (rosenbrock 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Expected Improvement per unit cost with exploration bonus.
    
    mu: (2000,) GP mean predictions
    sigma: (2000,) GP standard deviation predictions
    f_best: scalar, best observed value
    cost: (2000,) cost of evaluating each candidate
    
    Returns: (2000,) acquisition scores (higher = more promising)
    """
    # Deadline guard: ensure we return within time limits
    start_time = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized distance
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip negative EI to zero (shouldn't happen mathematically but for safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: EI per unit cost
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Base score: EI / cost
    score = ei / cost_safe
    
    # Add exploration bonus for high uncertainty regions
    # This helps escape local optima and explore the space better
    # Use a small weight to balance exploitation vs exploration
    exploration_weight = 0.1
    exploration_bonus = exploration_weight * sigma_safe / cost_safe
    
    score = score + exploration_bonus
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 280:  # Safety margin before 300s limit
        return score
    
    return score
# EVOLVE-BLOCK-END
