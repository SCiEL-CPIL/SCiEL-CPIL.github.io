"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure x is a 1D array for consistent operations
    x = np.asarray(x)
    
    # Define a diverse set of basis functions to capture different function shapes
    # 1. Polynomial terms up to x^3
    b0 = np.ones_like(x)
    b1 = x
    b2 = x**2
    b3 = x**3
    
    # 2. Trigonometric terms
    b4 = np.sin(x)
    b5 = np.cos(x)
    
    # 3. Hyperbolic tangent (saturation)
    b6 = np.tanh(x)
    
    # 4. Exponential terms (decay and growth)
    b7 = np.exp(-x**2) # Gaussian bump
    b8 = np.exp(-x)    # Decay
    b9 = np.exp(x)     # Growth
    
    # Combine linearly with parameters
    # y = p0*b0 + p1*b1 + ... + p9*b9
    result = (
        params[0] * b0 +
        params[1] * b1 +
        params[2] * b2 +
        params[3] * b3 +
        params[4] * b4 +
        params[5] * b5 +
        params[6] * b6 +
        params[7] * b7 +
        params[8] * b8 +
        params[9] * b9
    )
    
    return result
# EVOLVE-BLOCK-END
