import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # 5 seconds before the 330s limit


def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Return a safe default (e.g., uniform random or zero)
        return np.zeros_like(mu)
    
    # Cost-aware Expected Improvement with exploration bonus
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Small exploration bonus for uncertain regions
    exploration = 0.05 * sigma
    return (ei + exploration) / (cost + 1e-6)
