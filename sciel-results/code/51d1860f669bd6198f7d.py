"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model.
    
    Features:
    - Constant bias
    - Linear position (x) and velocity (v)
    - Cubic position (x^3) and velocity (v^3) for Duffing/Van der Pol-like nonlinearity
    - Quadratic velocity (v^2)
    - Cross terms: x*v and x^2*v (Van der Pol damping)
    - Sinusoidal driving: sin(t), cos(t)
    
    10 parameters total.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute common terms
    x2 = x * x
    v2 = v * v
    x3 = x2 * x
    v3 = v2 * v
    x2v = x2 * v
    
    # Model:
    # p0: bias
    # p1: linear stiffness
    # p2: linear damping
    # p3: cubic stiffness (Duffing)
    # p4: cubic damping
    # p5: quadratic damping
    # p6: cross linear (x*v)
    # p7: Van der Pol damping (x^2*v)
    # p8: sine forcing
    # p9: cosine forcing
    
    result = (
        p0
        + p1 * x
        + p2 * v
        + p3 * x3
        + p4 * v3
        + p5 * v2
        + p6 * x * v
        + p7 * x2v
        + p8 * np.sin(t)
        + p9 * np.cos(t)
    )
    
    return result
# EVOLVE-BLOCK-END
