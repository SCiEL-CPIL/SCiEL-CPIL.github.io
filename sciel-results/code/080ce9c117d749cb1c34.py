import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64)
    
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    # Strategy: Product of two low-degree polynomials
    # P(x) = p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4 + p5*x^5
    # Q(x) = p6 + p7*x + p8*x^2 + p9*x^3
    # Note: Q is degree 3 to keep total parameters at 10 (6+4).
    # This creates a polynomial of degree up to 8.
    # The non-convexity allows fitting sharper curves if the target function 
    # has features not well-captured by a simple linear combination of fixed basis functions.
    
    # Compute P(x)
    # Using Horner's method for numerical stability and speed
    p_poly = p[0]
    p_poly = p_poly * x + p[1]
    p_poly = p_poly * x + p[2]
    p_poly = p_poly * x + p[3]
    p_poly = p_poly * x + p[4]
    p_poly = p_poly * x + p[5]
    
    # Compute Q(x)
    q_poly = p[6]
    q_poly = q_poly * x + p[7]
    q_poly = q_poly * x + p[8]
    q_poly = q_poly * x + p[9]
    
    y = p_poly * q_poly
    
    return y
