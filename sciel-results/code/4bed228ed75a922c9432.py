"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible closed-form expression using polynomial, trigonometric,
    and exponential basis functions to recover an unknown function.
    
    The expression is designed to be flexible enough to fit a wide variety
    of functions while remaining differentiable for BFGS optimization.
    
    params: array of length 10
      p0, p1, p2, p3: polynomial coefficients (x, x^2, x^3, x^4)
      p4, p5: trigonometric coefficients (sin(x), cos(x))
      p6: exponential rate
      p7: exponential amplitude
      p8, p9: additional polynomial terms (x^5, x^6)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial basis
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    x6 = x4 * x2
    
    poly = p0 * x + p1 * x2 + p2 * x3 + p3 * x4 + p8 * x5 + p9 * x6
    
    # Trigonometric basis (periodic components)
    trig = p4 * np.sin(x) + p5 * np.cos(x)
    
    # Exponential basis (growth/decay components)
    # Clip to avoid overflow in exp
    exp_arg = np.clip(p6 * x, -20, 20)
    exp_term = p7 * np.exp(exp_arg)
    
    # Combine all terms
    return poly + trig + exp_term
# EVOLVE-BLOCK-END
