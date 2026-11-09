"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining Adaptive UCB and Expected Improvement (EI).
    
    The score is a weighted sum of:
    - Adaptive UCB: -mu + beta * sigma, where beta adapts to f_best
    - EI: (f_best - mu) * Phi(z) + sigma * phi(z)
    
    Early on (high f_best), UCB dominates for exploration.
    Late on (low f_best), EI dominates for exploitation.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(mu)
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Safety checks
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        return np.zeros_like(mu)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-10)
    
    # Adaptive beta based on f_best
    f_min = 0.397887
    f_max_est = 15.0
    
    f_current = np.clip(f_best, f_min, f_max_est)
    t = (f_current - f_min) / (f_max_est - f_min)
    
    beta_min = 0.5
    beta_max = 3.0
    beta = beta_min + (beta_max - beta_min) * t
    
    # UCB score for minimization: -mu + beta * sigma
    ucb_score = -mu + beta * sigma
    
    # EI score for minimization
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    ei_score = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Normalize both scores to have zero mean and unit variance for combination
    ucb_mean = np.mean(ucb_score)
    ucb_std = np.std(ucb_score)
    if ucb_std > 1e-10:
        ucb_norm = (ucb_score - ucb_mean) / ucb_std
    else:
        ucb_norm = ucb_score
    
    ei_mean = np.mean(ei_score)
    ei_std = np.std(ei_score)
    if ei_std > 1e-10:
        ei_norm = (ei_score - ei_mean) / ei_std
    else:
        ei_norm = ei_score
    
    # Blend: more UCB early (high t), more EI late (low t)
    # weight_ucb = t (1 early, 0 late), weight_ei = 1 - t
    weight_ucb = t
    weight_ei = 1.0 - t
    
    # Combine
    scores = weight_ucb * ucb_norm + weight_ei * ei_norm
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
