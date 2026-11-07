"""Rewritten program: hybrid UCB+EI acquisition with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # 300s limit, keep 20s margin


def _check_deadline():
    return time.time() < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Hybrid UCB + Expected Improvement acquisition function.
    
    Parameters:
    - mu: (2000,) array of GP mean predictions
    - sigma: (2000,) array of GP std dev predictions
    - f_best: float, best observed value so far
    - cost: (2000,) array of evaluation costs
    
    Returns:
    - (2000,) array of acquisition scores (higher = more promising)
    """
    # Deadline guard: if we're running out of time, return a safe fallback
    if not _check_deadline():
        # Return a simple, fast score to avoid timeout
        return -mu / (cost + 1e-9)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute UCB (Upper Confidence Bound) - for minimization, we want low values
    # UCB = -mu + beta * sigma (we want to minimize, so negative mu is good)
    beta = 2.0
    ucb = -mu + beta * sigma_safe
    
    # Combine: blend EI and UCB, both normalized by cost
    # EI is already positive when mu < f_best, so it's a good "improvement" signal
    # UCB is a balance of exploitation (low mu) and exploration (high sigma)
    
    # Normalize EI and UCB to comparable scales
    ei_cost = ei / (cost + 1e-9)
    ucb_cost = ucb / (cost + 1e-9)
    
    # Weighted combination: 0.5 EI + 0.5 UCB
    score = 0.5 * ei_cost + 0.5 * ucb_cost
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
