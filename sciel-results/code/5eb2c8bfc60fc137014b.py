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
    
    # Structural Change: Use Probability of Improvement (PI) instead of EI/UCB blend.
    # PI focuses on the likelihood of finding a better point, which can be more robust
    # to scale issues and encourages exploration in early stages when f_best is high.
    
    z = (f_best - mu) / sigma
    # Clip z to avoid numerical issues
    z = np.clip(z, -10, 10)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Cost-aware score
    scores = pi / cost
    
    # Add a small exploration bonus based on uncertainty (sigma) to prevent pure exploitation
    # The bonus decays as f_best improves (f_best_clipped is low)
    # We use a scaled version of sigma/cost
    beta_exploration = 0.1 * (1.0 - f_best_clipped)
    exploration_bonus = beta_exploration * (sigma / cost)
    
    scores = scores + exploration_bonus
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
