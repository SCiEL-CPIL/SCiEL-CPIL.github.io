"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Compute basis functions
    # Scaling polynomial terms to keep magnitudes O(1) for better numerical conditioning
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    # Scaled polynomials
    b0 = 1.0
    b1 = x
    b2 = x2 / 3.0
    b3 = x3 / 9.0
    b4 = x4 / 27.0
    
    # Trigonometric terms (already bounded)
    b5 = np.sin(x)
    b6 = np.cos(x)
    b7 = np.sin(2.0 * x)
    b8 = np.cos(2.0 * x)
    
    # Gaussian bump (bounded by 1)
    b9 = np.exp(-x2)
    
    # Combine
    return (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
# EVOLVE-BLOCK-END
