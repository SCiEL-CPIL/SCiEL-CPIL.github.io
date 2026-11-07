"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition: draw a posterior sample and pick the
    candidate that maximizes the sampled value per unit cost.
    
    mu, sigma: shape (2000,), posterior mean and std in objective units.
    f_best: lowest observed value (scalar).
    cost: shape (2000,), evaluation cost for each candidate.
    
    Returns: array of 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard
    t_start = time.time()
    deadline = 300.0 - 5.0  # safety margin
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw a single standard normal sample for each candidate
    z = np.random.standard_normal(len(mu))
    
    # Posterior sample: f_s = mu + sigma * z
    f_s = mu + sigma_safe * z
    
    # Cost-aware score: maximize f_s / (cost - 0.5)
    # We subtract 0.5 from cost to account for the base cost of 1 and the
    # fact that cost = 1 + mean(x_scaled), so cost is in [1, 2].
    # Using (cost - 0.5) as denominator ensures we're dividing by a positive
    # number in [0.5, 1.5].
    denom = np.maximum(cost - 0.5, 0.1)
    
    scores = f_s / denom
    
    # Replace any non-finite values with a large negative number
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
