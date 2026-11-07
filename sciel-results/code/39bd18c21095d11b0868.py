"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression combining polynomial, trigonometric, and hyperbolic
    basis functions to recover an unknown function.
    
    Parameters (10 total):
        p0: constant
        p1: linear
        p2: quadratic
        p3: cubic
        p4: quartic
        p5: sin coefficient
        p6: cos coefficient
        p7: tanh coefficient
        p8: exp coefficient (with clipped argument)
        p9: rational-like coefficient
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial basis (up to degree 4)
    poly = p0 + p1 * x + p2 * x**2 + p3 * x**3 + p4 * x**4
    
    # Trigonometric basis for periodic components
    trig = p5 * np.sin(x) + p6 * np.cos(x)
    
    # Hyperbolic tangent for saturating behavior
    tanh_term = p7 * np.tanh(x)
    
    # Exponential term with clipped argument to avoid overflow
    exp_arg = np.clip(x, -5, 5)
    exp_term = p8 * np.exp(exp_arg)
    
    # Rational-like smooth term: 1/(1 + x^2) scaled
    rational = p9 / (1.0 + x**2)
    
    return poly + trig + tanh_term + exp_term + rational
# EVOLVE-BLOCK-END
