"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 300.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() - _start_time > _TIMEOUT - _SAFETY_MARGIN:
        # Return a safe, finite score if we are about to timeout
        # This shouldn't happen if the function is fast, but it's a safety net.
        return np.zeros(len(mu))

    # Cost-Aware UCB (Upper Confidence Bound for minimization)
    # We want to maximize the score.
    # The "optimistic" lower bound for minimization is mu - xi * sigma.
    # We want this value to be low.
    # So we negate it to make it a maximization problem.
    # Score = -(mu - xi * sigma) / cost
    
    # Tuning xi: 
    # xi=0.5 was too exploratory (score 0.84).
    # xi=0.0 is pure exploitation (mu).
    # Let's try xi=0.2 for a balance.
    xi = 0.2
    
    # Calculate the UCB value (lower bound on function value)
    ucb_values = mu - xi * sigma
    
    # We want to minimize ucb_values.
    # So the acquisition score should be high for low ucb_values.
    # Score = -ucb_values / cost
    
    # Ensure cost is not zero (min cost is 1, so this is safe)
    safe_cost = np.maximum(cost, 1e-6)
    
    scores = -ucb_values / safe_cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
