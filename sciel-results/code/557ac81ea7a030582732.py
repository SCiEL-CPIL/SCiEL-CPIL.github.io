"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _deadline_guard():
    """Return remaining time in seconds with a safety margin."""
    return 280.0 - (time.time() - _start_time) if _start_time else 280.0


_start_time = None


def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: if we're running out of time, return a simple cost-based score
    remaining = _deadline_guard()
    if remaining < 5.0:
        # Fallback: simple cost-aware exploitation
        return np.where(cost > 0, 1.0 / (cost + 1e-9), 0.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Cost-aware Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei_cost = ei / (cost + 1e-9)
    
    # Cost-aware UCB (Upper Confidence Bound)
    # Beta controls exploration-exploitation tradeoff
    # Use a time-adaptive beta: start more exploratory, become more exploitative
    # We can't easily know the iteration, but we can use the spread of costs
    # as a proxy for how much exploration has happened
    beta = 1.0  # Fixed beta for stability
    
    ucb = mu + beta * sigma
    ucb_cost = ucb / (cost + 1e-9)
    
    # Cost-aware Probability of Improvement (PI)
    pi = norm.cdf((f_best - mu) / sigma)
    pi_cost = pi / (cost + 1e-9)
    
    # Blend the three acquisition functions
    # Weighting: EI is the primary driver, UCB adds exploration, PI adds robustness
    w_ei = 0.5
    w_ucb = 0.3
    w_pi = 0.2
    
    score = w_ei * ei_cost + w_ucb * ucb_cost + w_pi * pi_cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
