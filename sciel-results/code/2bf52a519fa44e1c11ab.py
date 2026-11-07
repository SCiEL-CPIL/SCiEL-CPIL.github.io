import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model using a polynomial expansion.
    Includes linear and cubic position terms, linear and quadratic velocity terms,
    and interaction terms.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial terms
    # Position terms
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    # Velocity terms
    v2 = v * v
    v3 = v2 * v
    
    # Interaction terms
    xv = x * v
    x2v = x2 * v
    
    # Construct the equation
    # We use up to 10 params.
    # p0: x
    # p1: x^2
    # p2: x^3
    # p3: v
    # p4: v^2
    # p5: v^3
    # p6: x*v
    # p7: x^2*v
    # p8: x^4
    # p9: constant
    
    return (p0 * x + 
            p1 * x2 + 
            p2 * x3 + 
            p3 * v + 
            p4 * v2 + 
            p5 * v3 + 
            p6 * xv + 
            p7 * x2v + 
            p8 * x4 + 
            p9)
