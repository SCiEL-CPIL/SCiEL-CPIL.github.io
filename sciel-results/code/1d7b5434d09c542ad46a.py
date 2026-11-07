"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 300.0  # Seconds


def acquisition(mu, sigma, f_best):
    # Structural Change: Revert to standard Expected Improvement (EI)
    # The previous version added an arbitrary exploration bonus (0.1 * sigma)
    # which likely degraded performance by distorting the acquisition landscape.
    # Pure EI is the standard baseline and achieved high scores in inspirations.
    
    # Safety check: If we are extremely close to the deadline,
    # return a simple exploitation score to avoid timeout.
    if time.time() > _DEADLINE - 1.0:
        return -mu
    
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure numerical stability
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
