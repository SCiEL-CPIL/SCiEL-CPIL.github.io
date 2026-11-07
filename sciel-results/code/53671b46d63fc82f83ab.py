import time
import numpy as np
from scipy.stats import norm

_T0 = time.perf_counter()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best, cost):
    if time.perf_counter() - _T0 > _DEADLINE:
        return 1.0 / np.maximum(cost, 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Increased max alpha to 2.5 to penalize cost more heavily in late game
    # Transition from 1.0 to 2.5 as f_best goes from 1.0 to 0.0
    alpha = 1.0 + 1.5 * np.clip(1.0 - f_best, 0.0, 1.0)
    
    scores = ei / np.power(cost_safe, alpha)
    
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
