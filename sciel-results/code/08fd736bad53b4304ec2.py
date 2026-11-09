"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: 9th-order Power Basis Polynomial.
    
    y(x) = sum_{i=0}^{9} p[i] * x^i
    
    This is a structural change from the previous Chebyshev+Gaussian hybrid.
    It uses all 10 parameters to define a single smooth polynomial curve.
    This family is universal for smooth functions on compact domains.
    """
    # Compute powers of x
    # Using np.power or repeated multiplication for stability
    # x^0, x^1, ..., x^9
    
    result = np.zeros_like(x)
    x_power = np.ones_like(x)
    
    for i in range(10):
        result += params[i] * x_power
        x_power = x_power * x
        
    return result
# EVOLVE-BLOCK-END
