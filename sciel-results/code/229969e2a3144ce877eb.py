"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Polynomial basis model for nonlinear oscillator acceleration.
    Uses a full 3rd-order polynomial in scaled position (x) and velocity (v).
    This captures linear and nonlinear restoring forces and damping terms.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Scale inputs to roughly [-1, 1] range for better numerical conditioning
    # x range: [-0.2056, 0.1749], center ~ -0.015, half-range ~ 0.19
    # v range: [-0.3391, 0.3624], center ~ 0.011, half-range ~ 0.35
    # Using simple scaling around 0
    xs = x / 0.20
    vs = v / 0.35

    # Polynomial terms up to degree 3
    # Order: 1, x, v, x^2, xv, v^2, x^3, x^2v, xv^2, v^3
    
    term_0 = 1.0
    term_1 = xs
    term_2 = vs
    term_3 = xs * xs
    term_4 = xs * vs
    term_5 = vs * vs
    term_6 = term_3 * xs
    term_7 = term_3 * vs
    term_8 = term_5 * xs
    term_9 = term_5 * vs

    # Combine with parameters
    # p0: bias
    # p1: linear x
    # p2: linear v
    # p3: quadratic x
    # p4: cross x v
    # p5: quadratic v
    # p6: cubic x
    # p7: cubic x v (x^2 v)
    # p8: cubic v x (x v^2)
    # p9: cubic v
    
    return (p0 * term_0 + 
            p1 * term_1 + 
            p2 * term_2 + 
            p3 * term_3 + 
            p4 * term_4 + 
            p5 * term_5 + 
            p6 * term_6 + 
            p7 * term_7 + 
            p8 * term_8 + 
            p9 * term_9)
# EVOLVE-BLOCK-END
