"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Global Polynomial Basis Expansion (Degree 3).
    
    Structure:
    - Basis functions: All monomials x^i * y^j where i+j <= 3.
    - Coefficients: 10 parameters (p0..p9) corresponding to the 10 basis functions.
    
    Basis Functions (10 total):
    1. 1
    2. x
    3. y
    4. x^2
    5. x*y
    6. y^2
    7. x^3
    8. x^2*y
    9. x*y^2
    10. y^3
    
    Model:
    f(x, y) = p0 + p1*x + p2*y + p3*x^2 + p4*x*y + p5*y^2 
             + p6*x^3 + p7*x^2*y + p8*x*y^2 + p9*y^3
    
    This approach is structurally different from RBFs:
    - RBFs are localized; Polynomials are global.
    - Polynomials are excellent for smooth functions defined on bounded domains.
    - The design matrix is linear in parameters, allowing BFGS to solve it efficiently.
    
    Parameters (10 total):
    p0: coeff for 1
    p1: coeff for x
    p2: coeff for y
    p3: coeff for x^2
    p4: coeff for x*y
    p5: coeff for y^2
    p6: coeff for x^3
    p7: coeff for x^2*y
    p8: coeff for x*y^2
    p9: coeff for y^3
    """
    if params is None or len(params) < 10:
        return np.zeros_like(x)
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis terms
    # We can vectorize this for speed, though N=200 is small.
    # However, constructing the full matrix each time is robust for BFGS.
    
    # To ensure numerical stability and speed, we compute the terms directly.
    # Note: x and y are arrays of shape (N,)
    
    # Term 1: 1
    t1 = 1.0
    
    # Term 2: x
    t2 = x
    
    # Term 3: y
    t3 = y
    
    # Term 4: x^2
    t4 = x * x
    
    # Term 5: x*y
    t5 = x * y
    
    # Term 6: y^2
    t6 = y * y
    
    # Term 7: x^3
    t7 = t4 * x
    
    # Term 8: x^2*y
    t8 = t4 * y
    
    # Term 9: x*y^2
    t9 = t5 * y
    
    # Term 10: y^3
    t10 = t6 * y
    
    # Combine
    # Using Horner's method or simple linear combination.
    # Simple linear combination is clear and fast enough for N=200.
    
    result = (p0 * t1 + 
              p1 * t2 + 
              p2 * t3 + 
              p3 * t4 + 
              p4 * t5 + 
              p5 * t6 + 
              p6 * t7 + 
              p7 * t8 + 
              p8 * t9 + 
              p9 * t10)
              
    return result
# EVOLVE-BLOCK-END
