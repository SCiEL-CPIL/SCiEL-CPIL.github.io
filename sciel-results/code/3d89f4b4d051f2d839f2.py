"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluates the model using a rich basis of polynomial, trigonometric,
    hyperbolic, and Gaussian functions.
    params[0..9] are coefficients for:
    1, x, x^2, x^3, x^4, x^5, sin(x), cos(x), tanh(x), exp(-x^2)
    """
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis functions
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    tanh_x = np.tanh(x)
    exp_neg_x2 = np.exp(-x2)
    
    # Combine
    y = (p0 * 1.0 +
         p1 * x +
         p2 * x2 +
         p3 * x3 +
         p4 * x4 +
         p5 * x5 +
         p6 * sin_x +
         p7 * cos_x +
         p8 * tanh_x +
         p9 * exp_neg_x2)
         
    return y
# EVOLVE-BLOCK-END
