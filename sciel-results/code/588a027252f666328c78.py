"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Universal function approximator using 10 parameters.
    
    Strategy: Use a combination of:
    1. A polynomial basis (up to degree 4) - 5 params
    2. A periodic component (sin and cos) - 2 params
    3. A rational component (1/(x^2 + a)) - 1 param
    4. A linear and constant term - 2 params
    
    Total: 10 parameters
    
    The key insight is that many common functions can be approximated well by:
    - Polynomials (for smooth, non-oscillatory behavior)
    - Trigonometric terms (for periodic behavior)
    - Rational terms (for functions with singularities or decay)
    """
    # Extract parameters
    p = params.copy()
    
    # Ensure numerical stability
    # Clip parameters to avoid overflow
    p = np.clip(p, -1000, 1000)
    
    # Polynomial basis: 1, x, x^2, x^3, x^4
    # Coefficients: p[0], p[1], p[2], p[3], p[4]
    poly = (p[0] + p[1] * x + p[2] * x**2 + p[3] * x**3 + p[4] * x**4)
    
    # Trigonometric basis: sin(x), cos(x)
    # Coefficients: p[5], p[6]
    # Use a frequency parameter derived from p[5] and p[6] to allow different frequencies
    # Actually, let's keep it simple: sin(x) and cos(x) with fixed frequency 1
    trig = p[5] * np.sin(x) + p[6] * np.cos(x)
    
    # Rational component: 1 / (x^2 + a) where a = |p[7]| + 0.1
    # Coefficient: p[8]
    a = np.abs(p[7]) + 0.1
    rational = p[8] / (x**2 + a)
    
    # Additional linear and constant for fine-tuning
    # Coefficients: p[9]
    # Let's add a tanh component for bounded functions
    tanh_term = p[9] * np.tanh(x)
    
    result = poly + trig + rational + tanh_term
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
