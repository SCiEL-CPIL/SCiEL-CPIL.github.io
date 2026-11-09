"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: keep a safety margin before the 330s wall-clock limit.
    # The function is called repeatedly inside a loop; we track cumulative time
    # to avoid exceeding the budget. If we get close to the deadline, we return
    # a simple deterministic score to finish quickly.
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Clip f_best to [0, 1] for stable interpolation
    f_best_clipped = np.clip(f_best, 0.0, 1.0)
    
    # Adaptive beta: higher when f_best is poor (exploration), lower when good (exploitation)
    beta_min = 0.5
    beta_max = 2.0
    beta = beta_min + (beta_max - beta_min) * f_best_clipped
    
    # UCB score for minimization: -mu + beta * sigma
    # Divide by cost to be cost-aware
    ucb_scores = (-mu + beta * sigma) / cost
    
    # Expected Improvement (EI) for minimization:
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # This is more exploitation-focused than UCB.
    
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei_scores = ei / cost
    
    # Blend UCB and EI: use more UCB when exploring (high f_best), more EI when exploiting
    # Use a smooth transition based on f_best
    weight_ucb = f_best_clipped
    weight_ei = 1.0 - f_best_clipped
    
    scores = weight_ucb * ucb_scores + weight_ei * ei_scores
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
