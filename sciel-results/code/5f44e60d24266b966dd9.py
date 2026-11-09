import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE_START = time.time()
_TIME_LIMIT = 310.0  # Safety margin before 330s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _DEADLINE_START)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (hartmann 3d).
    Uses a blend of Expected Improvement (EI) and Probability of Improvement (PI)
    with cost-awareness.
    """
    # Check deadline
    if _time_left() < 0.5:
        return np.zeros_like(mu)
    
    # Ensure finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute EI for minimization
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    ei = np.maximum(ei, 0.0)
    
    # Compute PI (Probability of Improvement) for minimization
    z_pi = (f_best - mu) / sigma_safe
    pi = norm.cdf(z_pi)
    
    # Normalize EI and PI to [0, 1]
    ei_max = np.max(ei) if np.max(ei) > 1e-10 else 1.0
    ei_norm = ei / ei_max
    
    pi_max = np.max(pi) if np.max(pi) > 1e-10 else 1.0
    pi_norm = pi / pi_max
    
    # Blend EI and PI
    # Weight PI more for exploration, EI for exploitation
    alpha = 0.6  # weight for EI
    beta = 0.4   # weight for PI
    
    blended = alpha * ei_norm + beta * pi_norm
    
    # Cost-aware adjustment: divide by cost to prefer cheaper evaluations
    scores = blended / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
