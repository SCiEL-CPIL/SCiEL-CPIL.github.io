"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Complete cubic polynomial basis: 10 terms spanning all monomials up to degree 3."""
    p = params
    return (p[0] + 
            p[1]*x + p[2]*y + 
            p[3]*x**2 + p[4]*y**2 + p[5]*x*y + 
            p[6]*x**3 + p[7]*y**3 + p[8]*x**2*y + p[9]*x*y**2)
# EVOLVE-BLOCK-END
