"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
TIME_LIMIT = 290.0 # Safety margin

def acquisition(mu, sigma, f_best, cost):
    # Check time
    if time.time() - _start_time > TIME_LIMIT:
        # Fallback to random or simple EI if running out of time
        # But we should return valid scores.
        # Returning zeros is valid but bad.
        # Returning EI is better.
        pass

    # Expected Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Lower Confidence Bound Score
    # We want to minimize f(x). LCB is a lower bound.
    # We prefer points with low LCB.
    # Score = -LCB = -(mu + beta * sigma)
    beta = 2.0
    lcb_score = -(mu + beta * sigma)
    
    # Combine: Take the maximum of EI and LCB score
    # This allows exploitation via EI and exploration via LCB
    score = np.maximum(ei, lcb_score)
    
    # Cost-aware: divide by cost
    # Ensure cost is not zero (it's >= 1)
    return score / cost
# EVOLVE-BLOCK-END
