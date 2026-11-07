"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track time for safety margin
_start_time = time.time()
TIME_LIMIT = 320  # seconds, with 10s safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining UCB exploration with EI-like exploitation.
    
    Uses a hybrid approach:
    - When uncertainty (sigma) is high relative to improvement potential, 
      favor exploration (UCB-like)
    - When improvement potential is clear, favor exploitation (EI-like)
    
    Returns higher scores for points more likely to improve f_best.
    """
    global _start_time
    
    # Check time budget
    elapsed = time.time() - _start_time
    if elapsed > TIME_LIMIT:
        # Return a safe fallback score
        return -mu + sigma
    
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute expected improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute UCB score (for minimization: -mu + beta*sigma)
    # Use adaptive beta based on how far we are from convergence
    # When f_best is close to 0 (minimum), be more exploitative
    # When f_best is far from 0, be more exploratory
    if f_best > 0.1:
        beta = 2.5  # More exploration when far from optimum
    elif f_best > 0.01:
        beta = 1.5  # Moderate exploration
    else:
        beta = 0.5  # More exploitation when close to optimum
    
    ucb = -mu + beta * sigma_safe
    
    # Normalize both scores to [0, 1] range for combination
    ei_min, ei_max = np.min(ei), np.max(ei)
    ei_range = ei_max - ei_min + 1e-10
    ei_norm = (ei - ei_min) / ei_range
    
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    ucb_range = ucb_max - ucb_min + 1e-10
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Combine: weight EI more when we have good improvement signals,
    # weight UCB more when uncertainty is high
    # Use a dynamic weighting based on the ratio of EI to UCB variability
    ei_var = np.var(ei_norm) + 1e-10
    ucb_var = np.var(ucb_norm) + 1e-10
    
    # Higher EI variance means EI is providing more discrimination
    # Higher UCB variance means UCB is providing more discrimination
    w_ei = ei_var / (ei_var + ucb_var)
    w_ucb = 1.0 - w_ei
    
    score = w_ei * ei_norm + w_ucb * ucb_norm
    
    # Scale back to a reasonable range
    score = score * 10.0
    
    return score
# EVOLVE-BLOCK-END
