"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erf
import time

_START = time.time()
_DEADLINE = _START + 325.0

_SQRT_2PI = np.sqrt(2.0 * np.pi)
_INV_SQRT_2 = 1.0 / np.sqrt(2.0)


def acquisition(mu, sigma, f_best):
    """Numerically stable Expected Improvement for minimization."""
    # Deadline guard: if we're running out of time, return a safe fallback
    if time.time() > _DEADLINE:
        # Pure exploitation fallback
        scores = f_best - mu
        return np.where(np.isfinite(scores), scores, 0.0)

    sigma_safe = np.maximum(sigma, 1e-8)
    delta = f_best - mu  # positive when mu < f_best (improvement possible)
    
    z = delta / sigma_safe
    
    # Compute phi(z) and Phi(z) using raw formulas for speed and stability
    # phi(z) = exp(-0.5 * z^2) / sqrt(2*pi)
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    
    # Avoid overflow in z^2 by clipping z before squaring
    # If z is very large, phi(z) is 0, Phi(z) is 1 (or 0)
    # erf is bounded, so Phi is safe. exp underflows to 0, which is safe.
    # But z^2 can overflow if z > ~1e154.
    # Clipping z to [-10, 10] is safe and sufficient for double precision.
    z_clipped = np.clip(z, -10.0, 10.0)
    
    phi_z = np.exp(-0.5 * z_clipped * z_clipped) / _SQRT_2PI
    Phi_z = 0.5 * (1.0 + erf(z_clipped * _INV_SQRT_2))
    
    ei = delta * Phi_z + sigma_safe * phi_z
    
    # Ensure non-negative (EI should be >= 0)
    ei = np.maximum(ei, 0.0)
    
    # Ensure finiteness
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
# EVOLVE-BLOCK-END
