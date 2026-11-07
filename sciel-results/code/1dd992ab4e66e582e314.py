"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Chebyshev polynomials (0-5) + sine/cosine terms.
    This captures both polynomial and periodic components.
    """
    t = x / 2.0  # Map to [-1, 1]
    
    # Chebyshev polynomials T_0 through T_5
    T = np.zeros((6, len(x)))
    T[0, :] = 1.0
    T[1, :] = t
    for n in range(2, 6):
        T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
    
    # Sine and cosine terms (using original x for natural frequency)
    sin_terms = np.sin(x)
    cos_terms = np.cos(x)
    
    # Combine: 6 Chebyshev + 4 trig = 10 parameters total
    result = np.zeros_like(x)
    for i in range(6):
        result += params[i] * T[i, :]
    result += params[6] * sin_terms
    result += params[7] * cos_terms
    result += params[8] * np.sin(2 * x)
    result += params[9] * np.cos(2 * x)
    
    return result
# EVOLVE-BLOCK-END
