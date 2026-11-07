import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_TIMEOUT = 330.0  # Seconds


def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _START_TIME > _TIMEOUT - 1.0:
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Structural Change: Pure LCB with moderate exploration constant
    # Previous program used k=3.0 and switched to EI.
    # Inspiration 2 (high score) used UCB (equiv to LCB for minimization) with k=1.0.
    # LCB: -mu + k * sigma. Higher is better (more likely to be min).
    
    sigma_safe = np.maximum(sigma, 1e-8)
    k = 1.0
    
    score = -mu + k * sigma_safe
    
    # Sanitize
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    score = np.clip(score, -1e6, 1e6)

    return score
