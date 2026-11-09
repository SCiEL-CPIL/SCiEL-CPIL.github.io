"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization.
    # We want to minimize f.
    # LCB = mu - beta * sigma
    # We select the point with the lowest LCB.
    # The interface requires higher score to be selected.
    # Score = -LCB = -mu + beta * sigma
    
    # Choose beta to balance exploration and exploitation.
    # A fixed beta is used as step count is not available.
    beta = 1.5
    
    # Ensure sigma is positive to avoid division issues or negative variance effects, 
    # though sigma should be positive from GP.
    # If sigma is 0, UCB becomes -mu.
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    # If any are NaN or Inf, replace them with a very low score (since higher is better, 
    # we want to avoid selecting them, so set to -inf or a very negative number? 
    # Actually, if sigma is huge, score is huge. If mu is huge negative, score is huge positive.
    # The contract says "Return 2000 finite scores".
    # So we must clip or replace non-finite values.
    
    # If a score is +Inf, it's good? No, if sigma is Inf, it's bad.
    # Usually GP sigma is finite.
    # Let's just ensure finiteness.
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    return scores
# EVOLVE-BLOCK-END
