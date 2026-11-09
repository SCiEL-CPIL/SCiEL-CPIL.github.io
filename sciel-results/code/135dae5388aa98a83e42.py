import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # 5s margin before 330s limit

def acquisition(mu, sigma, f_best):
    # Deadline guard: if we are too close to the limit, return a safe fallback
    # Since this function is very fast, this is unlikely to trigger, but it's required.
    if time.time() > _DEADLINE:
        # Fallback to LCB with beta=0 (just -mu) or similar to avoid hanging
        return -mu + 0.1 * sigma

    # Structural change: Switch from fixed LCB to Expected Improvement (EI)
    # EI is a non-linear acquisition function that adapts better to the posterior
    # distribution than the linear LCB.
    
    # Ensure sigma is strictly positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Return finite scores
    return ei
