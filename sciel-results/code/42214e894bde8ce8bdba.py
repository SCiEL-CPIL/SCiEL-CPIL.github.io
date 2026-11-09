# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # Safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a reasonable fallback: pure EI/cost without extra computation
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        score = ei / cost_safe
        return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    cost = np.maximum(cost, 1e-9)
    
    # Calculate z.
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Calculate Expected Improvement.
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Use a modest exploration bonus to balance exploitation and exploration.
    # A smaller bonus (0.05) keeps focus on expected improvement while still
    # allowing some exploration of high-uncertainty regions.
    exploration_bonus = 0.05 * sigma
    
    # Calculate the score.
    score = (ei + exploration_bonus) / cost
    
    # Ensure the scores are finite.
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
