import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a 3rd-order polynomial basis.
    The basis includes all monomials x^i * v^j where i + j <= 3.
    This allows for Duffing-like terms (x^3), drag-like terms (xv, v^2), etc.
    """
    p = params
    # Map parameters to polynomial terms
    # Degree 0
    c0 = p[0]
    # Degree 1
    c1x = p[1]
    c1v = p[2]
    # Degree 2
    c2xx = p[3]
    c2xv = p[4]
    c2vv = p[5]
    # Degree 3
    c3xxx = p[6]
    c3xxv = p[7]
    c3xvv = p[8]
    c3vvv = p[9]
    
    # Compute powers
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Construct the expression
    # Using Horner-like or direct summation. Direct is fine for 10 terms.
    term0 = c0
    term1 = c1x * x + c1v * v
    term2 = c2xx * x2 + c2xv * x * v + c2vv * v2
    term3 = c3xxx * x3 + c3xxv * x2 * v + c3xvv * x * v2 + c3vvv * v3
    
    return term0 + term1 + term2 + term3
