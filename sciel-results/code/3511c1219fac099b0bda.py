"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Approach: Rational Function (Ratio of two polynomials of degree 4)
    # y = (p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4) / (q0 + q1*x + q2*x^2 + q3*x^3 + q4*x^4)
    # Parameters:
    # 0-4: p0, p1, p2, p3, p4 (Numerator coefficients)
    # 5-9: q0, q1, q2, q3, q4 (Denominator coefficients)
    
    p = params[:5]
    q = params[5:]
    
    # Compute powers of x
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    # Numerator
    num = p[0] + p[1] * x + p[2] * x2 + p[3] * x3 + p[4] * x4
    
    # Denominator
    den = q[0] + q[1] * x + q[2] * x2 + q[3] * x3 + q[4] * x4
    
    # Safe division
    # Add a tiny epsilon to prevent division by zero if optimizer drives den to 0
    # However, for smooth functions, den shouldn't be 0.
    # To be safe, we can clip the denominator away from 0.
    # But clipping changes the gradient.
    # Let's just do standard division, but handle potential overflow/zero.
    # If den is very small, the value explodes.
    # Given the initialization, den is positive.
    
    return num / den
# EVOLVE-BLOCK-END
