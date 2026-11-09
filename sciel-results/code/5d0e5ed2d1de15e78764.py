"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START = time.time()
_DEADLINE = _START + 325.0


def acquisition(mu, sigma, f_best):
    """Numerically stable Expected Improvement for minimization."""
    # Deadline guard: if we're running out of time, return a safe fallback
    if time.time() > _DEADLINE:
        # Pure exploitation fallback
        scores = f_best - mu
        return np.where(np.isfinite(scores), scores, 0.0)

    sigma_safe = np.maximum(sigma, 1e-12)
    delta = f_best - mu  # positive when mu < f_best (improvement possible)

    # Numerically stable EI:
    # For z > 0: EI = delta * Phi(z) + sigma * phi(z)
    # For z < 0: EI = delta * (1 - Phi(-z)) + sigma * phi(-z)  [equivalent but avoids issues]
    # Actually the standard formula works fine with scipy, but let's guard against extreme values.
    
    z = delta / sigma_safe
    
    # Clip z to avoid extreme values that could cause numerical issues
    z_clipped = np.clip(z, -10, 10)
    
    ei = delta * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Ensure non-negative (EI should be >= 0)
    ei = np.maximum(ei, 0.0)
    
    # Ensure finiteness
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
