"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores using Gaussian Process UCB (Upper Confidence Bound)
    for minimization.
    
    UCB score = -mu + beta * sigma
    
    This balances exploitation (low predicted mean mu) and exploration (high
    uncertainty sigma). The beta parameter controls the trade-off: larger beta
    favors exploration, smaller beta favors exploitation.
    
    We use a moderate beta value that should work well for the Ackley function
    in early-to-mid optimization stages.
    """
    # Deadline guard: 330s limit. Return immediately if close to limit.
    # 10s safety margin.
    if time.time() - _START_TIME > 320:
        return np.zeros_like(mu)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-12)
    
    # UCB parameter: controls exploration vs exploitation trade-off
    # A value around 2-4 is common for GP-UCB
    beta = 2.0
    
    # UCB score: we want to maximize this, so we negate mu (since we minimize)
    # and add the confidence bound
    scores = -mu + beta * sigma
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
