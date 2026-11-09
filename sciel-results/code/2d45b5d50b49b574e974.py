import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a complete polynomial basis up to degree 2
    in 3 variables.
    
    Basis functions:
    1. 1
    2. x
    3. y
    4. z
    5. x^2
    6. y^2
    7. z^2
    8. xy
    9. xz
    10. yz
    
    This uses all 10 parameters and provides a flexible model that can
    capture linear trends, curvature, and pairwise interactions.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis terms
    b0 = 1.0
    b1 = x
    b2 = y
    b3 = z
    b4 = x * x
    b5 = y * y
    b6 = z * z
    b7 = x * y
    b8 = x * z
    b9 = y * z
    
    # Linear combination
    pred = (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3 + 
            p4 * b4 + p5 * b5 + p6 * b6 + p7 * b7 + 
            p8 * b8 + p9 * b9)
            
    return pred
