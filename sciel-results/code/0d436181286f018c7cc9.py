"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # 10-feature basis: polynomial + nonlinear interaction
    # Features: 1, x, y, x^2, y^2, x*y, x^3, y^3, x^2*y, tanh(x*y)
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    y2 = y * y
    xy = x * y
    
    return (p0 + 
            p1 * x + 
            p2 * y + 
            p3 * x2 + 
            p4 * y2 + 
            p5 * xy + 
            p6 * x2 * x + 
            p7 * y2 * y + 
            p8 * x2 * y + 
            p9 * np.tanh(xy))
# EVOLVE-BLOCK-END
