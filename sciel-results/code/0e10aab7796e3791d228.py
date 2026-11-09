import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # 5s margin before 330s limit

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback to LCB with beta=0 (just -mu) or similar to avoid hanging
        return -mu + 0.1 * sigma

    # Ensure sigma is strictly positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Probability of Improvement (PI)
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This favors points where there is a high probability of finding a value
    # lower than the current best.
    
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Return finite scores
    return pi
