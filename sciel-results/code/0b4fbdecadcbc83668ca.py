"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = None


def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: if we're running long, return simple EI/cost
    elapsed = time.time() - _start_time
    if elapsed > 280:  # Safety margin before 300s limit
        z = (f_best - mu) / (sigma + 1e-9)
        return ((f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)) / cost
    
    # Cost-aware acquisition: blend EI/cost with exploration
    # EI per unit cost
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei_cost = ei / cost
    
    # UCB-style exploration term, also cost-aware
    # Beta controls exploration vs exploitation
    beta = 0.5  # Moderate exploration
    ucb_cost = (mu + beta * sigma) / cost
    
    # Blend: weight EI/cost more when f_best is good (exploit), 
    # weight UCB/cost more when uncertain (explore)
    # Use a smooth blend based on how much improvement EI suggests
    ei_mean = np.mean(ei)
    if ei_mean > 1e-6:
        # Strong improvement signal -> exploit
        w = 0.8
    else:
        # Weak improvement signal -> explore more
        w = 0.5
    
    score = w * ei_cost + (1 - w) * ucb_cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
