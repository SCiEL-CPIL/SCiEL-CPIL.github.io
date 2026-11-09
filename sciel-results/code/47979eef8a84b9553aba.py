# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = None

def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    # Inflation factor for sigma to encourage exploration
    # This effectively creates a "soft" EI that is more exploratory
    # than standard EI.
    inflation = 2.0
    sigma_eff = sigma * inflation
    
    # Avoid division by zero
    sigma_safe = np.where(sigma_eff > 1e-9, sigma_eff, 1e-9)
    
    # Compute Expected Improvement with inflated sigma
    z = (f_best - mu) / sigma_safe
    # EI = (f_best - mu) * Phi(z) + sigma_eff * phi(z)
    # Note: using sigma_eff in the second term makes the exploration reward larger
    ei = (f_best - mu) * norm.cdf(z) + sigma_eff * norm.pdf(z)
    
    # Ensure finite output
    score = np.where(np.isfinite(ei), ei, 0.0)
    
    return score
# EVOLVE-BLOCK-END
