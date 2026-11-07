# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_DEADLINE = None
_DEADLINE_LIMIT = 310.0  # Safety margin before 330s kill

def _init_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _DEADLINE_LIMIT

def _check_deadline():
    global _DEADLINE
    if _DEADLINE is not None and time.time() > _DEADLINE:
        return True
    return False

def acquisition(mu, sigma, f_best, cost):
    _init_deadline()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and cost is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Structural change: Expected Improvement (EI) acquisition
    # EI = E[max(0, f_best - f(x))]
    # For f(x) ~ N(mu, sigma^2):
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    
    diff = f_best - mu
    z = diff / sigma_safe
    z = np.clip(z, -10, 10)
    
    # Phi(z) is CDF, phi(z) is PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = diff * Phi_z + sigma_safe * phi_z
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    scores = ei / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    if _check_deadline():
        return scores
    
    return scores
# EVOLVE-BLOCK-END
