"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: polynomial (degree 5) + two Gaussian bumps.
    params[0:6]  -> polynomial coefficients for (x/3)^k, k=0..5
    params[6:8]  -> Gaussian widths (absolute value used)
    params[8:10] -> Gaussian amplitudes
    Fixed Gaussian centers at x = -1.5 and x = 1.5.
    """
    # Scale x to [-1, 1] for polynomial conditioning
    xs = x / 3.0
    
    # Polynomial basis (6 terms)
    result = np.zeros_like(x, dtype=np.float64)
    xs_power = np.ones_like(x, dtype=np.float64)
    for k in range(6):
        result += params[k] * xs_power
        xs_power *= xs
    
    # Two Gaussian bumps with fixed centers
    c1, c2 = -1.5, 1.5
    w1 = max(abs(params[6]), 0.1)
    w2 = max(abs(params[7]), 0.1)
    a1 = params[8]
    a2 = params[9]
    
    result += a1 * np.exp(-((x - c1) / w1) ** 2)
    result += a2 * np.exp(-((x - c2) / w2) ** 2)
    
    return result
# EVOLVE-BLOCK-END
