# EVOLVE-BLOCK-START
import numpy as np


def rudin_problem10() -> np.ndarray:
    """
    Returns the coefficients for the zonal spherical harmonic of degree 10 (m=0).
    This harmonic is the unique axisymmetric harmonic of degree 10.
    Based on the directive to exploit rotational symmetry, this is the candidate.
    """
    k = 10
    coeffs = np.zeros(2 * k + 1, dtype=np.float64)
    
    # The zonal harmonic corresponds to m = 0.
    # Index for m=0 is k.
    coeffs[k] = 1.0
    
    return coeffs


# EVOLVE-BLOCK-END
