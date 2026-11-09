"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track start time for deadline enforcement
_START_TIME = time.time()
_DEADLINE_SECONDS = 325.0  # Safety margin before 330s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,) with posterior means in objective units
    - sigma: array of shape (2000,) with posterior standard deviations in objective units
    - f_best: lowest observed value so far
    
    Returns:
    - array of shape (2000,) with scores; higher means more promising
    """
    # Check deadline - if we're close to timeout, return a simple fallback
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        # Fallback: return negative mu (pure exploitation) to finish quickly
        return -mu
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) for minimization
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # Use stable computation to avoid overflow/underflow
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = improvement * cdf + sigma_safe * pdf
    
    # UCB term for exploration
    # Adaptive beta: start with more exploration, decrease over time
    # We estimate progress from f_best relative to typical Beale function values
    # Beale min is 0, max is ~22516 on [-4.5,4.5]^2
    # Use a simple heuristic: if f_best is still large, explore more
    if f_best > 100:
        beta = 2.0
    elif f_best > 10:
        beta = 1.5
    elif f_best > 1:
        beta = 1.0
    else:
        beta = 0.5
    
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and UCB: use EI as primary, UCB as secondary
    # Normalize both to have similar scale
    ei_scale = np.std(ei) if np.std(ei) > 1e-9 else 1.0
    ucb_scale = np.std(ucb) if np.std(ucb) > 1e-9 else 1.0
    
    ei_norm = ei / ei_scale
    ucb_norm = ucb / ucb_scale
    
    # Weighted combination: 70% EI, 30% UCB
    score = 0.7 * ei_norm + 0.3 * ucb_norm
    
    # Handle potential numerical issues
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
