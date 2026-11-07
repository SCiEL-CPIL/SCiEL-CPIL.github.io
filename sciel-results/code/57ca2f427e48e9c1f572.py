"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Center the inputs to improve numerical conditioning for BFGS
    # Input range is [0.5, 2], so center is 1.25
    cx = x - 1.25
    cy = y - 1.25
    cz = z - 1.25
    
    # Full quadratic polynomial basis in centered variables
    # 10 terms:
    # 0: constant
    # 1: cx
    # 2: cy
    # 3: cz
    # 4: cx^2
    # 5: cy^2
    # 6: cz^2
    # 7: cx*cy
    # 8: cx*cz
    # 9: cy*cz
    
    p = params
    
    return (
        p[0]
        + p[1] * cx
        + p[2] * cy
        + p[3] * cz
        + p[4] * cx**2
        + p[5] * cy**2
        + p[6] * cz**2
        + p[7] * cx * cy
        + p[8] * cx * cz
        + p[9] * cy * cz
    )
# EVOLVE-BLOCK-END
