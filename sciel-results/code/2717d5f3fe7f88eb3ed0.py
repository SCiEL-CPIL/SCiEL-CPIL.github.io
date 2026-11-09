"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track deadline across calls within a single evaluation run
_start_time = None
_DEADLINE_FRACTION = 0.95  # Safety margin: stop at 95% of 330s

def _get_elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _reset_timer():
    global _start_time
    _start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining EI and UCB with adaptive exploration.
    
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP std predictions
    f_best: lowest observed value
    
    Returns: (2000,) array of scores, higher = more promising
    """
    global _start_time
    
    # If this is the first call, reset the timer
    if _start_time is None:
        _reset_timer()
    
    elapsed = _get_elapsed()
    deadline = 330.0 * _DEADLINE_FRACTION
    
    # Safety guard: if we're close to deadline, return a simple fallback
    if elapsed > deadline:
        return -mu  # Exploit: pick the point with lowest predicted mean
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)  # Numerical stability
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # UCB component: explore regions with high uncertainty
    # The beta parameter controls exploration vs exploitation
    # We use an adaptive beta that starts high and decreases
    n_evals_estimate = int(elapsed / 0.05)  # Rough estimate of evaluations done
    # Start with beta ~ 2.0, decay to ~ 0.5
    beta = 0.5 + 1.5 * np.exp(-n_evals_estimate / 10.0)
    
    # UCB score
    ucb = -mu + beta * sigma_safe
    
    # Combine: weighted sum of EI and UCB
    # EI is already in "improvement" units, UCB is in "value + uncertainty" units
    # We normalize to make them comparable
    # Scale EI by typical improvement scale
    ei_scale = max(np.std(f_best - mu), 1e-10)
    ucb_scale = max(np.std(sigma_safe), 1e-10)
    
    # Normalize both to have similar magnitude
    ei_norm = ei / (ei_scale + 1e-10)
    ucb_norm = ucb / (ucb_scale + 1e-10)
    
    # Weighted combination: 60% EI, 40% UCB
    score = 0.6 * ei_norm + 0.4 * ucb_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
