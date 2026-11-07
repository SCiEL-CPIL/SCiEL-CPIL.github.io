import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model.
    
    Basis:
    1. Constant (bias)
    2. x (linear restoring)
    3. x^2 (asymmetric restoring)
    4. x^3 (nonlinear stiffness, Duffing)
    5. v (linear damping)
    6. v^2 (quadratic damping)
    7. v^3 (cubic damping)
    8. sin(t) (periodic driving)
    9. cos(t) (periodic driving)
    10. x*v (cross term)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute basis functions
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    xv = x * v
    
    # Combine
    return (p0 + 
            p1 * x + 
            p2 * x2 + 
            p3 * x3 + 
            p4 * v + 
            p5 * v2 + 
            p6 * v3 + 
            p7 * sin_t + 
            p8 * cos_t + 
            p9 * xv)
