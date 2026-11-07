"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter basis mixing trigonometric (Fourier-like) terms and polynomial terms.
    Uses cos/sin for x and y to capture periodic or oscillatory behavior,
    plus low-order polynomial terms for smooth non-periodic components.
    
    Basis: 1, x, y, x^2, y^2, cos(x), sin(x), cos(y), sin(y), x*y
    
    This structure can model:
    - Periodic functions via cos/sin terms
    - Smooth polynomial trends via 1, x, y, x^2, y^2
    - Interaction effects via x*y
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial terms
    x2 = x * x
    y2 = y * y
    xy = x * y
    
    # Trigonometric terms
    cos_x = np.cos(x)
    sin_x = np.sin(x)
    cos_y = np.cos(y)
    sin_y = np.sin(y)
    
    return (p0 * 1 + p1 * x + p2 * y + p3 * x2 + p4 * y2 +
            p5 * cos_x + p6 * sin_x + p7 * cos_y + p8 * sin_y +
            p9 * xy)
# EVOLVE-BLOCK-END
