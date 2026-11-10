import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a product of two polynomials.
    P(x) of degree 5 and Q(x) of degree 4, resulting in a degree 9 polynomial.
    params[0:6] are coefficients for P(x) (a0 to a5)
    params[6:10] are coefficients for Q(x) (b0 to b4)
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    a = params[0:6]
    b = params[6:10]
    
    # Compute powers of x
    # We need x^0 to x^5 for P and x^0 to x^4 for Q
    # Max power needed is 5
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    
    # Build P(x) = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5
    p_val = a[0] + a[1] * x + a[2] * x2 + a[3] * x3 + a[4] * x4 + a[5] * x5
    
    # Build Q(x) = b0 + b1*x + b2*x^2 + b3*x^3 + b4*x^4
    q_val = b[0] + b[1] * x + b[2] * x2 + b[3] * x3 + b[4] * x4
    
    # Return product
    return p_val * q_val
