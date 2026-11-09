"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-params basis function expansion with a richer basis set.
    params: array of length 10
    
    The basis functions are designed to span a wide variety of common functional
    forms: polynomials up to degree 5, trigonometric functions with multiple
    frequencies, exponential decay/growth, and hyperbolic functions.
    
    Since the model is linear in params, BFGS should converge to the global
    optimum for the training data.
    """
    # Extract 10 coefficients
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
    
    # Polynomial basis
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    
    # Trigonometric basis
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2.0 * x)
    cos_2x = np.cos(2.0 * x)
    sin_3x = np.sin(3.0 * x)
    
    # Exponential basis (bounded to avoid overflow)
    exp_x = np.exp(np.clip(x, -5, 5))
    exp_neg_x = np.exp(np.clip(-x, -5, 5))
    
    # Hyperbolic tangent basis
    tanh_x = np.tanh(x)
    tanh_2x = np.tanh(2.0 * x)
    
    # Absolute value and sign-like smooth approximations
    abs_x = np.abs(x)
    smooth_sign = np.tanh(3.0 * x)
    
    # Combine all basis functions
    result = (
        c0 * np.ones_like(x) +
        c1 * x +
        c2 * x2 +
        c3 * x3 +
        c4 * x4 +
        c5 * sin_x +
        c6 * cos_x +
        c7 * tanh_x +
        c8 * exp_x +
        c9 * exp_neg_x
    )
    
    return result
# EVOLVE-BLOCK-END
