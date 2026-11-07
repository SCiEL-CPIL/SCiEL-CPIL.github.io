import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    Uses a complete polynomial basis of degree 3.
    """
    p = params
    
    # Complete polynomial basis of degree 3
    # 1. Constant
    # 2. x
    # 3. v
    # 4. x^2
    # 5. x*v
    # 6. v^2
    # 7. x^3
    # 8. x^2*v
    # 9. x*v^2
    # 10. v^3
    
    return (
        p[0] + 
        p[1] * x + 
        p[2] * v + 
        p[3] * x**2 + 
        p[4] * x * v + 
        p[5] * v**2 + 
        p[6] * x**3 + 
        p[7] * x**2 * v + 
        p[8] * x * v**2 + 
        p[9] * v**3
    )
