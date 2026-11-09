"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import ndtr, ndtri
import math

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s kill

def acquisition(mu, sigma, f_best):
    """
    Adaptive UCB acquisition for 1D sphere minimization.
    
    Uses a UCB-style score: score = -mu + beta * sigma
    where beta adapts based on how close f_best is to the known minimum (0).
    
    - When f_best is far from 0: higher beta (more exploration)
    - When f_best is close to 0: lower beta (more exploitation)
    
    This directly optimizes for low predicted values while allowing
    uncertainty-driven exploration early on.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: simple UCB with fixed beta
        sigma_safe = np.maximum(sigma, 1e-12)
        score = -mu + 0.1 * sigma_safe
        return np.where(np.isfinite(score), score, 0.0)

    # Numerical safety for sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Adaptive beta: scale exploration based on current best
    # f_best is the lowest observed value; for sphere, min is 0
    # When f_best is large (far from optimum), we need MORE exploration
    # When f_best is small (close to optimum), we can exploit more
    # Normalize f_best: max possible is 25 (at x=±5)
    f_norm = min(f_best / 25.0, 1.0)
    # Beta ranges from ~0.02 (exploit) to ~0.20 (explore)
    beta = 0.02 + 0.18 * f_norm
    
    # UCB score for minimization: higher score = more likely to be good
    # We want low mu and high sigma (uncertainty might hide a better value)
    score = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
