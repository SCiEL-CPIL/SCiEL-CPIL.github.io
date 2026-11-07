import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a nonlinear oscillator: full cubic polynomial in x and v."""
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Basis: [x, v, x², x³, v², v³, x*v, x²*v, x*v², 1]
    return (p0 * x + p1 * v + p2 * x2 + p3 * x3 + p4 * v2 + p5 * v3 + 
            p6 * x * v + p7 * x2 * v + p8 * x * v2 + p9)
