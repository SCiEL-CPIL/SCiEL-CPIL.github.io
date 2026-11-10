# EVOLVE-BLOCK-START
import numpy as np


def rudin_problem10() -> np.ndarray:
    """
    Returns the coefficients for the zonal spherical harmonic of degree 10 (m=0).
    This harmonic is expected to minimize the L4/L2 norm ratio compared to the
    all-ones baseline, as it is more "spread out" than sectorial harmonics and
    less "noisy" than the all-ones sum.
    """
    k = 10
    # Initialize coefficients to zero
    coeffs = np.zeros(2 * k + 1, dtype=np.float64)
    
    # The zonal harmonic corresponds to m = 0.
    # The index for m is i, where m = i - k.
    # So for m = 0, i = k = 10.
    coeffs[k] = 1.0
    
    return coeffs


# EVOLVE-BLOCK-END
