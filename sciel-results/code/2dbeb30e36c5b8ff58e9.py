"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Expected Improvement per unit cost with adaptive exploration.
    # Uses a time-aware guard to ensure we stay within the 300s limit.
    start_time = time.time()
    
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to favor cheaper evaluations
    base_score = ei / cost
    
    # Adaptive exploration: add a small UCB term to encourage exploration
    # when uncertainty is high, but keep it modest to not disrupt exploitation
    # The exploration coefficient is scaled by the median sigma to be robust
    median_sigma = np.median(sigma_safe)
    if median_sigma > 1e-9:
        # Use a small beta value for exploration
        beta = 0.1 * median_sigma
        ucb_term = beta * sigma_safe
        score = base_score + ucb_term
    else:
        score = base_score
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard: if we've used too much time, return a simple result
    elapsed = time.time() - start_time
    if elapsed > 280:  # Safety margin before 300s limit
        return base_score
    
    return score
# EVOLVE-BLOCK-END
