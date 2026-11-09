"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns prediction using a hybrid basis combining:
    - Legendre polynomials (degree 0-5) for smooth polynomial components
    - Trigonometric functions for periodic components
    - Gaussian for localized components
    - tanh for saturation components
    
    This hybrid basis is designed to cover common symbolic regression targets
    while maintaining good conditioning for BFGS optimization.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Scale input to [-1, 1] for better conditioning
    t = x / 3.0
    
    n = x.shape[0]
    
    # Build basis functions
    # First 6: Legendre polynomials P_0 through P_5
    # These are orthogonal w.r.t. uniform weight, matching the input distribution
    
    # P_0(t) = 1
    b0 = np.ones(n, dtype=np.float64)
    
    # P_1(t) = t
    b1 = t.copy()
    
    # P_2(t) = (3t^2 - 1)/2
    b2 = (3.0 * t**2 - 1.0) / 2.0
    
    # P_3(t) = (5t^3 - 3t)/2
    b3 = (5.0 * t**3 - 3.0 * t) / 2.0
    
    # P_4(t) = (35t^4 - 30t^2 + 3)/8
    b4 = (35.0 * t**4 - 30.0 * t**2 + 3.0) / 8.0
    
    # P_5(t) = (63t^5 - 70t^3 + 15t)/8
    b5 = (63.0 * t**5 - 70.0 * t**3 + 15.0 * t) / 8.0
    
    # Next 4: Trigonometric functions for periodic components
    # Using frequencies that match the domain [-1, 1]
    b6 = np.sin(np.pi * t)
    b7 = np.cos(np.pi * t)
    b8 = np.sin(2.0 * np.pi * t)
    b9 = np.cos(2.0 * np.pi * t)
    
    # Combine with parameters
    y = (params[0] * b0 + 
         params[1] * b1 + 
         params[2] * b2 + 
         params[3] * b3 + 
         params[4] * b4 + 
         params[5] * b5 + 
         params[6] * b6 + 
         params[7] * b7 + 
         params[8] * b8 + 
         params[9] * b9)
    
    return y
# EVOLVE-BLOCK-END
