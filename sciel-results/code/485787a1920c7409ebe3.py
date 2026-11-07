"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0  # 300s limit, use 300s for safety margin? 
# The evaluator kills at 330s. We should finish well before.
# Each call is fast.

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() > _DEADLINE - 5.0:
        # Return a safe fallback: Greedy exploitation
        return -mu
    
    # Use UCB with beta = 1.0
    # Minimization: Score = -mu + beta * sigma
    beta = 1.0
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
