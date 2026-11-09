import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator (Duffing type) with driving force.
    
    Model: a = c0 + c1*x + c2*x^2 + c3*x^3 + c4*v + c5*v^2 + c6*v^3 + c7*x*v + c8*x^2*v + c9*v^2*x
    
    This captures:
    - Linear restoring (c1*x)
    - Nonlinear restoring/Duffing (c2*x^2, c3*x^3)
    - Linear damping (c4*v)
    - Nonlinear damping (c5*v^2, c6*v^3)
    - Coupling terms (c7*x*v, c8*x^2*v, c9*v^2*x)
    """
    p = params
    
    # Precompute powers
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # 10 basis terms (monomial basis, physically motivated)
    y = (p[0] +
         p[1] * x +
         p[2] * x2 +
         p[3] * x3 +
         p[4] * v +
         p[5] * v2 +
         p[6] * v3 +
         p[7] * x * v +
         p[8] * x2 * v +
         p[9] * x * v2)
    
    return y
