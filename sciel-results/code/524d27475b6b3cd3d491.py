"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a diverse basis set that includes
    polynomials, trigonometric, exponential, and hyperbolic functions.
    
    The model is: y = sum_{i=0}^{9} p_i * phi_i(x)
    where phi_i are chosen to span a wide variety of functional forms.
    
    params: array of length 10
    """
    # Extract parameters
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
    
    # Polynomial basis
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    
    # Trigonometric basis
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2.0 * x)
    
    # Exponential basis (clamped for numerical stability)
    exp_x = np.exp(np.clip(x, -15, 15))
    exp_neg_x = np.exp(-np.clip(x, -15, 15))
    
    # Hyperbolic basis
    tanh_x = np.tanh(x)
    
    # Combine all basis functions
    # c0: 1 (bias)
    # c1: x
    # c2: x^2
    # c3: x^3
    # c4: x^4
    # c5: sin(x)
    # c6: cos(x)
    # c7: sin(2x)
    # c8: tanh(x)
    # c9: exp(x) - exp(-x) = 2*sinh(x), scaled for stability
    
    # Use sinh-like combination for c9 to capture exponential growth/decay
    sinh_term = 0.5 * (exp_x - exp_neg_x)
    
    result = (
        c0 * np.ones_like(x) +
        c1 * x +
        c2 * x2 +
        c3 * x3 +
        c4 * x4 +
        c5 * sin_x +
        c6 * cos_x +
        c7 * sin_2x +
        c8 * tanh_x +
        c9 * sinh_term
    )
    
    return result
# EVOLVE-BLOCK-END
