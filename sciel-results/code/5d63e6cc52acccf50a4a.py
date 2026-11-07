# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter pure trigonometric (Fourier) basis.
    Replaces polynomial terms with higher-frequency sine/cosine terms and interaction terms.
    
    Basis:
    1. 1
    2. cos(x)
    3. sin(x)
    4. cos(y)
    5. sin(y)
    6. cos(2*x)
    7. sin(2*x)
    8. cos(2*y)
    9. sin(2*y)
    10. cos(x)*cos(y)
    
    This structure aims to capture periodic or oscillatory behaviors that polynomials
    approximate poorly, especially near the boundaries of [-1, 1] if the function
    has curvature or periodicity.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute trigonometric terms
    cos_x = np.cos(x)
    sin_x = np.sin(x)
    cos_y = np.cos(y)
    sin_y = np.sin(y)
    
    cos_2x = np.cos(2 * x)
    sin_2x = np.sin(2 * x)
    cos_2y = np.cos(2 * y)
    sin_2y = np.sin(2 * y)
    
    # Interaction term
    cos_x_cos_y = cos_x * cos_y
    
    return (p0 * 1 + p1 * cos_x + p2 * sin_x + p3 * cos_y + p4 * sin_y +
            p5 * cos_2x + p6 * sin_2x + p7 * cos_2y + p8 * sin_2y +
            p9 * cos_x_cos_y)
# EVOLVE-BLOCK-END
