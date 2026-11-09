"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're running too long, return a safe fallback
    # that prioritizes exploration (high sigma) to avoid getting stuck.
    # But since we don't have a global deadline tracker here, we keep it simple.
    
    # Expected improvement per unit cost, with a small exploration bonus
    # to avoid zero scores when sigma is very small.
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration term to encourage visiting uncertain regions
    # when EI is very small (e.g., near the best point)
    exploration = sigma * 0.1
    
    # Combine EI and exploration, then divide by cost
    score = (ei + exploration) / (cost + 1e-9)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
