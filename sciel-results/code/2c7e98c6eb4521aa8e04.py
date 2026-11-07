import numpy as np
from scipy.stats import norm
import time

_deadline_start = time.time()
_DEADLINE_LIMIT = 320.0

def _check_deadline():
    return (time.time() - _deadline_start) > _DEADLINE_LIMIT

def acquisition(mu, sigma, f_best, cost):
    if _check_deadline():
        return np.zeros(len(mu))
    
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-12)
    
    # EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive Parameters
    f_early = -3.0
    f_late = -3.8
    
    if f_best >= f_early:
        progress = 0.0
    elif f_best <= f_late:
        progress = 1.0
    else:
        progress = (f_early - f_best) / (f_early - f_late)
        progress = np.clip(progress, 0.0, 1.0)
        
    # Exploration weight (beta for UCB-like term)
    # High early, low late
    beta_max = 2.0
    beta = beta_max * (1.0 - progress)
    
    # Cost exponent
    # Low early, high late
    gamma = progress
    
    # Score
    exploration_term = beta * sigma_safe
    raw_score = ei + exploration_term
    
    cost_factor = np.power(cost_safe, gamma)
    scores = raw_score / cost_factor
    
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
