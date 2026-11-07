"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard
_DEADLINE = time.time() + 320.0  # 10s safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    mu, sigma: shape (2000,) - GP mean and std (in objective units)
    f_best: scalar - lowest observed value
    cost: shape (2000,) - evaluation cost for each candidate
    
    Returns: shape (2000,) - scores (higher = evaluate next)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative costs
    cost = np.maximum(cost, 1e-9)
    
    # Guard against zero/negative sigma
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute UCB for minimization: -mu + beta * sigma
    # We add f_best to make the baseline positive relative to improvement, 
    # though strictly UCB uses -mu + beta*sigma. 
    # Using (f_best - mu) aligns with EI's exploitation term but is linear.
    # Let's use standard UCB: score_base = -mu + beta * sigma
    # Since we want to maximize score, and lower mu is better:
    # score_base = -mu + beta * sigma
    
    # To ensure scores are positive and comparable, we can shift by f_best
    # or just rely on the ranking. 
    # Let's use: base = (f_best - mu) + beta * sigma
    # This is similar to EI but without the non-linearity of the normal CDF/PDF.
    
    beta = 2.0
    base = (f_best - mu) + beta * sigma
    
    # Structural Change: Non-linear cost penalty
    # The cost is in [1, 2]. 
    # Using cost**2 penalizes expensive points (cost=2) by factor 4 vs cost=1.
    # This is a stronger preference for cheap points than 1/cost.
    cost_penalty = cost ** 2
    
    score = base / cost_penalty
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
