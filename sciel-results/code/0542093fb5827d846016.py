import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE_START = time.time()
_TIME_LIMIT = 315.0  # Safety margin before 330s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _DEADLINE_START)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (hartmann 3d).
    Uses a structural blend of Expected Improvement (EI) and a dynamic UCB
    with adaptive exploration, both normalized and cost-adjusted.
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
    
    # Compute EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB for minimization: score = kappa * sigma - mu
    # Dynamic kappa based on gap between f_best and min(mu)
    mu_range = np.ptp(mu) if np.ptp(mu) > 1e-9 else 1.0
    min_mu = np.min(mu)
    gap = f_best - min_mu
    
    # Base kappa with adjustment
    kappa_base = 1.5
    if gap > 1e-6:
        kappa = kappa_base + 1.0 * np.tanh(gap / (0.5 * mu_range + 1e-9))
    else:
        kappa = kappa_base * 0.5
    
    # Clip kappa to reasonable range
    kappa = np.clip(kappa, 0.5, 4.0)
    
    # UCB score (higher is better for minimization)
    score_ucb = (kappa * sigma_safe - mu) / cost_safe
    
    # Normalize EI and UCB to [0, 1] using min-max
    ei_max = np.max(ei) if np.max(ei) > 1e-10 else 1.0
    ei_norm = ei / ei_max
    
    ucb_max = np.max(score_ucb) if np.max(score_ucb) > 1e-10 else 1.0
    ucb_norm = score_ucb / ucb_max
    
    # Blend: weight EI more for exploitation, UCB for exploration
    # Use a slightly higher weight on EI since it directly measures expected improvement
    alpha = 0.7
    beta = 0.3
    
    blended = alpha * ei_norm + beta * ucb_norm
    
    # Cost-aware adjustment: divide by cost to prefer cheaper evaluations
    scores = blended / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time guard
    t_elapsed = time.time() - _DEADLINE_START
    if t_elapsed > _TIME_LIMIT - 5:
        return scores
        
    return scores
