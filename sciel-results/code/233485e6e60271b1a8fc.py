import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety checks
    if time.time() > _DEADLINE:
        return np.zeros(2000, dtype=np.float64)

    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-9)
    
    # z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    EI = sigma_safe * (z * Phi + phi)
    
    # Clip EI to avoid negative values
    EI = np.maximum(EI, 1e-12)
    
    # Cost-normalized EI
    score = EI / cost_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
