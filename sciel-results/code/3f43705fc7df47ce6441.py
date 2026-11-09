"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


_START_TIME = time.time()
_DEADLINE = 330 - 5  # Safety margin


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback to a simple greedy strategy if we are running out of time
        return -mu / (cost + 1e-9)
    
    # Cost-aware acquisition: combine expected improvement and UCB
    # with a cost penalty.
    
    # Expected improvement (EI) for minimization
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB for minimization: we want to minimize mu - beta * sigma
    # So the "score" for UCB is -mu + beta * sigma
    beta = 1.5
    ucb_score = -mu + beta * sigma
    
    # Combine EI and UCB, both normalized by cost
    # EI is already non-negative and measures expected improvement
    # UCB score can be negative or positive
    # Normalize both to have similar scales
    # Scale EI by a factor to make it comparable to UCB
    # A common approach: score = EI / cost + lambda * ucb_score / cost
    
    # Actually, let's use a weighted combination
    # The EI term is in objective units, UCB is also in objective units
    # Let's try: score = (ei + 0.5 * ucb_score) / cost
    # But ucb_score = -mu + beta*sigma, which has different sign convention
    
    # Better: use EI as the primary signal, and add exploration via UCB
    # score = ei / cost + gamma * sigma / cost
    # where gamma controls exploration
    
    # Let's use a tunable combination
    gamma = 0.3
    score = (ei + gamma * sigma) / (cost + 1e-9)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
