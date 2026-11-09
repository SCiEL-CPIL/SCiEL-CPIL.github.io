# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 330.0
_MARGIN = 5.0

def _elapsed():
    return time.time() - _start_time

def acquisition(mu, sigma, f_best):
    """
    Adaptive UCB acquisition function.
    
    Uses a decaying beta parameter to balance exploration and exploitation.
    Early in the search (large f_best), beta is high to explore.
    Late in the search (small f_best), beta is low to exploit.
    """
    if _elapsed() > _DEADLINE - _MARGIN:
        # Fallback to simple EI if time is running out
        eps = 1e-9
        sigma_safe = np.maximum(sigma, eps)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.where(np.isfinite(ei), ei, 0.0)
        return ei

    n = len(mu)
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Adaptive Beta
    # We want beta to be large when f_best is large (early) and small when f_best is small (late).
    # f_best starts around ~0.5-1.0 for Weierstrass and goes to 0.
    # Let's use a decay function.
    # beta = beta_max * (f_best / (f_best + decay_scale))
    # If f_best = 1.0, decay_scale = 0.1 -> beta = 2.0 * (1.0/1.1) ~ 1.8
    # If f_best = 0.1, decay_scale = 0.1 -> beta = 2.0 * (0.1/0.2) = 1.0
    # If f_best = 0.01, decay_scale = 0.1 -> beta = 2.0 * (0.01/0.11) ~ 0.18
    
    beta_max = 2.0
    decay_scale = 0.05
    
    f_best_pos = max(f_best, eps)
    beta = beta_max * (f_best_pos / (f_best_pos + decay_scale))
    
    # UCB for minimization:
    # We want to maximize the score.
    # Lower mu is better. Higher sigma is better for exploration.
    # Score = -mu + beta * sigma
    
    scores = -mu + beta * sigma_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
