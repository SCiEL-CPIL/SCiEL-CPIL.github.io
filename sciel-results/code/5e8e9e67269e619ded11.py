"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIME_LIMIT = 290.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    # Check time limit
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback to standard EI/cost to ensure valid output
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        score = ei / cost
        return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    # Cost-aware Upper Confidence Bound (UCB)
    # We minimize the function, so we prefer low mu and high sigma.
    # Score = (-mu + kappa * sigma) / cost
    # Higher score indicates a better candidate.
    kappa = 1.5
    
    score = (-mu + kappa * sigma) / cost
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
