"""Symbolic regression: product of two low-degree polynomials."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: P5(x) * Q4(x) where P5 has 6 coefficients and Q4 has 5 coefficients.
    Total 11 parameters, but we use 10 by fixing one coefficient.
    
    Actually, let's use P5(x) * Q3(x) = 6 + 4 = 10 parameters.
    P5(x) = p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4 + p5*x^5
    Q3(x) = q0 + q1*x + q2*x^2 + q3*x^3
    
    Product: sum_{i=0}^{5} sum_{j=0}^{3} p_i * q_j * x^(i+j)
    This gives a polynomial of degree up to 8, with 10 parameters defining it
    in a structured (rank-1 coefficient matrix) way.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Extract coefficients
    p0, p1, p2, p3, p4, p5 = p[0], p[1], p[2], p[3], p[4], p[5]
    q0, q1, q2, q3 = p[6], p[7], p[8], p[9]
    
    # Compute P5(x)
    P5 = p0 + p1 * x + p2 * x**2 + p3 * x**3 + p4 * x**4 + p5 * x**5
    
    # Compute Q3(x)
    Q3 = q0 + q1 * x + q2 * x**2 + q3 * x**3
    
    return P5 * Q3
# EVOLVE-BLOCK-END
