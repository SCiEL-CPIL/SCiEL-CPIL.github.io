"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter symbolic regression model.
    
    Basis functions chosen for:
    - Good conditioning on [-1,1]^2
    - Coverage of polynomial, trigonometric, and interaction terms
    - Smooth gradients for BFGS optimization
    """
    p = params
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Polynomial terms (up to degree 3)
    t0 = x
    t1 = y
    t2 = x * y
    t3 = x * x
    t4 = y * y
    t5 = x * x * x
    t6 = y * y * y
    
    # Trigonometric terms (periodic behavior)
    t7 = np.sin(np.pi * x)
    t8 = np.sin(np.pi * y)
    
    # Constant
    t9 = np.ones_like(x)
    
    f = (p[0] * t0 + p[1] * t1 + p[2] * t2 + p[3] * t3 +
         p[4] * t4 + p[5] * t5 + p[6] * t6 + p[7] * t7 +
         p[8] * t8 + p[9] * t9)
    
    return f
# EVOLVE-BLOCK-END
