"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: kill after 330s wall-clock, use safety margin
_DEADLINE = 330.0
_START_TIME = time.time()
_TIMEOUT_MARGIN = 5.0  # seconds before deadline to stop

def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    return (time.time() - _START_TIME) > (_DEADLINE - _TIMEOUT_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining UCB and EI.
    
    mu, sigma: shape (2000,) - GP mean and std dev in objective units
    f_best: float - lowest observed value
    cost: shape (2000,) - evaluation cost for each candidate
    
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Check deadline
    if _check_deadline():
        # Return best exploitation signal as fallback
        return -(mu / np.maximum(cost, 1e-9))
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute UCB with adaptive exploration
    # Use a fixed beta that balances exploration/exploitation
    beta = 0.5
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and UCB, both normalized by cost
    # Weight: use EI primarily, add UCB for exploration
    score_ei = ei / np.maximum(cost, 1e-9)
    score_ucb = ucb / np.maximum(cost, 1e-9)
    
    # Blend: 70% EI, 30% UCB for balanced exploration
    combined = 0.7 * score_ei + 0.3 * score_ucb
    
    # Ensure finite output
    combined = np.where(np.isfinite(combined), combined, 0.0)
    
    return combined
# EVOLVE-BLOCK-END
