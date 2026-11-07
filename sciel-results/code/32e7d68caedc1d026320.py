"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series approximation with 10 parameters.
    params[0]: constant term (DC)
    params[1:10]: coefficients for sin/cos terms up to frequency 5.
    
    f(x) = c0 + c1*sin(x) + c2*cos(x) + c3*sin(2x) + c4*cos(2x) + ...
    
    This approach uses a trigonometric basis which is well-suited for
    smooth functions on a bounded domain and avoids the ill-conditioning
    issues of high-degree polynomials. The model is linear in parameters,
    ensuring robust convergence for the BFGS optimizer.
    """
    # Start with the constant term
    result = np.full_like(x, params[0], dtype=float)
    
    # Add sinusoidal components
    # We have 9 remaining parameters for 9 terms.
    # Let's use frequencies 1, 2, 3, 4, 5.
    # Pattern: sin(1x), cos(1x), sin(2x), cos(2x), sin(3x), cos(3x), sin(4x), cos(4x), sin(5x)
    
    result += params[1] * np.sin(x)
    result += params[2] * np.cos(x)
    
    result += params[3] * np.sin(2 * x)
    result += params[4] * np.cos(2 * x)
    
    result += params[5] * np.sin(3 * x)
    result += params[6] * np.cos(3 * x)
    
    result += params[7] * np.sin(4 * x)
    result += params[8] * np.cos(4 * x)
    
    result += params[9] * np.sin(5 * x)
    
    return result
# EVOLVE-BLOCK-END
