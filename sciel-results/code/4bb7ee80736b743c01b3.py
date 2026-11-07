import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model using a cubic polynomial basis in state (x, v).
    
    Model:
    a = p1 + p2*x + p3*v + p4*x^2 + p5*v^2 + p6*x*v + p7*x^3 + p8*v^3 + p9*x^2*v + p10*x*v^2
    
    This approach assumes the dynamics are autonomous (do not explicitly depend on time)
    and uses a complete monomial basis up to degree 3 to capture nonlinearities.
    """
    p1 = params[0]
    p2 = params[1]
    p3 = params[2]
    p4 = params[3]
    p5 = params[4]
    p6 = params[5]
    p7 = params[6]
    p8 = params[7]
    p9 = params[8]
    p10 = params[9]
    
    # Pre-compute powers
    x2 = x * x
    x3 = x2 * x
    
    v2 = v * v
    v3 = v2 * v
    
    # Cross terms
    xv = x * v
    x2v = x2 * v
    xv2 = x * v2
    
    # Construct the equation
    result = (
        p1 + 
        p2 * x + 
        p3 * v + 
        p4 * x2 + 
        p5 * v2 + 
        p6 * xv + 
        p7 * x3 + 
        p8 * v3 + 
        p9 * x2v + 
        p10 * xv2
    )
    
    return result
