"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: track elapsed time to ensure we return well before the 330s limit
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # 5s safety margin


def _check_deadline():
    """Return True if we should abort and return a safe fallback."""
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best):
    # If we're close to the deadline, return a simple UCB score as fallback
    if _check_deadline():
        return mu + 2.0 * sigma

    # Use Pure Expected Improvement (EI) for minimization
    # This is a structural change from the hybrid EI/UCB approach
    # EI is known to be effective for Bayesian optimization when the GP is well-calibrated
    
    sigma_safe = np.maximum(sigma, 1e-10)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite values
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
