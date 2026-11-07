"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rich basis with 10 parameters mixing polynomial and nonlinear terms.
    All basis functions are well-behaved on [-1,1].
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Polynomial terms
    x2 = x * x
    y2 = y * y
    xy = x * y
    
    # Nonlinear terms
    sin_x = np.sin(x)
    sin_y = np.sin(y)
    tanh_x = np.tanh(x)
    tanh_y = np.tanh(y)
    
    return (p0 + p1 * x + p2 * y + p3 * x2 + p4 * y2 + p5 * xy +
            p6 * sin_x + p7 * sin_y + p8 * tanh_x + p9 * tanh_y)
# EVOLVE-BLOCK-END
