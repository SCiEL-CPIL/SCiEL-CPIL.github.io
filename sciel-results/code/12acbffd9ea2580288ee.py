"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 320.0


def acquisition(mu, sigma, f_best):
    # Deadline guard: if we're running out of time, fall back to a simple heuristic
    if time.time() > _DEADLINE - 5.0:
        # Return a score that favors low mu (exploit) but with some exploration
        return -mu + 0.1 * sigma

    # Use Expected Improvement with a small exploration constant xi
    xi = 0.01
    s = np.maximum(sigma, 1e-8)
    z = (f_best - mu - xi) / s
    cdf_z = 0.5 * (1.0 + np.vectorize(__import__('math').erf)(z / np.sqrt(2.0)))
    pdf_z = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z * z)
    return (f_best - mu - xi) * cdf_z + s * pdf_z
# EVOLVE-BLOCK-END
