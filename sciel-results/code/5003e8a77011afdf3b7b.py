import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Polynomial basis model in raw inputs (x, v).
    Uses a complete polynomial of degree 2 in x and v, plus selected cubic terms.
    This gives the optimizer maximum flexibility to learn all scales and couplings.
    
    Basis terms (10 total):
    1. 1
    2. x
    3. v
    4. x^2
    5. x*v
    6. v^2
    7. x^3
    8. x^2*v
    9. x*v^2
    10. v^3
    """
    p = params
    
    # Compute all basis terms in raw units
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    xv = x * v
    x2v = x2 * v
    xv2 = x * v2
    
    # Linear combination of basis terms
    a = (p[0] + 
         p[1] * x + 
         p[2] * v + 
         p[3] * x2 + 
         p[4] * xv + 
         p[5] * v2 + 
         p[6] * x3 + 
         p[7] * x2v + 
         p[8] * xv2 + 
         p[9] * v3)
    
    return a
