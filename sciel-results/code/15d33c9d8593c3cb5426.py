"""
Symbolic regression with 2 inputs.
Uses a tensor-product Fourier basis with 10 terms.
This basis is linear in parameters, allowing stable convergence with BFGS.
The Fourier basis is orthogonal on [-1, 1], which improves conditioning
compared to monomial polynomial bases.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a tensor-product Fourier basis.
    The model is a linear combination of 10 Fourier terms:
    1, cos(pi*x), cos(pi*y), sin(pi*x), sin(pi*y),
    cos(pi*x)*cos(pi*y), sin(pi*x)*sin(pi*y),
    cos(2*pi*x), cos(2*pi*y), sin(2*pi*x)*sin(2*pi*y)
    
    This basis is orthogonal on [-1, 1] and linear in parameters,
    ensuring the optimization problem is convex and BFGS converges
    to the global minimum for this basis.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Pre-compute trigonometric functions
    cos_pi_x = np.cos(np.pi * x)
    sin_pi_x = np.sin(np.pi * x)
    cos_pi_y = np.cos(np.pi * y)
    sin_pi_y = np.sin(np.pi * y)
    cos_2pi_x = np.cos(2 * np.pi * x)
    cos_2pi_y = np.cos(2 * np.pi * y)
    sin_2pi_x = np.sin(2 * np.pi * x)
    sin_2pi_y = np.sin(2 * np.pi * y)
    
    # Construct prediction
    # Order: 1, cos(pi*x), cos(pi*y), sin(pi*x), sin(pi*y),
    #        cos(pi*x)*cos(pi*y), sin(pi*x)*sin(pi*y),
    #        cos(2*pi*x), cos(2*pi*y), sin(2*pi*x)*sin(2*pi*y)
    pred = (p0 
            + p1 * cos_pi_x 
            + p2 * cos_pi_y 
            + p3 * sin_pi_x 
            + p4 * sin_pi_y 
            + p5 * cos_pi_x * cos_pi_y 
            + p6 * sin_pi_x * sin_pi_y 
            + p7 * cos_2pi_x 
            + p8 * cos_2pi_y 
            + p9 * sin_2pi_x * sin_2pi_y)
            
    return pred
# EVOLVE-BLOCK-END
