# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_T0 = time.time()
_DEADLINE = _T0 + 325.0  # Safety margin vs 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    Uses a hybrid approach: primarily minimum variance (explore uncertainty)
    with a small exploitation component to avoid ignoring obvious improvements.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    fb = float(f_best)
    
    # Deadline check
    if time.time() > _DEADLINE - 1.0:
        return np.zeros(len(mu))
    
    s = np.maximum(sigma, 1e-8)
    
    # Minimum variance acquisition: score by posterior standard deviation
    # Higher sigma means more uncertainty, so we explore there
    min_var_score = s.copy()
    
    # Add a small exploitation component: prefer points where mu is low
    # This prevents pure exploration from ignoring obvious improvements
    # Scale mu contribution to be small relative to sigma
    mu_scale = np.max(s) / (np.max(np.abs(mu)) + 1e-8)
    exploit_score = -mu * mu_scale
    
    # Combine: primarily explore uncertainty, slightly exploit low mean
    scores = min_var_score + exploit_score
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
