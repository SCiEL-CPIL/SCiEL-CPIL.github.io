"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Dynamic UCB acquisition function for minimization.
    
    Uses an adaptive exploration coefficient that scales with the current
    best value and uncertainty levels to balance exploration and exploitation.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute the normalized uncertainty (coefficient of variation)
    # This helps adapt the exploration rate
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min
    if mu_range < 1e-10:
        mu_range = 1e-10
    
    sigma_mean = np.mean(sigma_safe)
    sigma_std = np.std(sigma_safe)
    
    # Dynamic UCB: the exploration coefficient adapts based on
    # how far f_best is from the best predicted mean
    # When f_best is much worse than mu_min, we explore more
    # When f_best is close to mu_min, we exploit more
    
    # Normalize f_best relative to the prediction range
    if f_best > mu_min:
        # Scale exploration: more exploration when f_best is far above mu_min
        gap = (f_best - mu_min) / (mu_range + 1e-10)
        # Clip gap to reasonable range
        gap = np.clip(gap, 0.1, 5.0)
        # Exploration coefficient: base + adaptive component
        beta = 1.0 + 2.0 * gap
    else:
        # f_best is already at or below mu_min, so exploit heavily
        beta = 0.5
    
    # Also scale by overall uncertainty level
    if sigma_mean > 1e-8:
        uncertainty_scale = np.clip(sigma_mean / (abs(f_best) + 1e-10), 0.1, 3.0)
        beta *= uncertainty_scale
    
    # Clip beta to reasonable range
    beta = np.clip(beta, 0.3, 10.0)
    
    # Compute UCB scores for minimization:
    # We want to minimize, so low values are good.
    # UCB for minimization: score = -(mu + beta * sigma)
    # Higher score means more promising (lower expected value with uncertainty bonus)
    scores = -(mu + beta * sigma_safe)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check deadline
    if time.time() > deadline:
        # Return best effort
        return scores
    
    return scores
# EVOLVE-BLOCK-END
