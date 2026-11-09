"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Basis functions selected to cover linear, interaction, and reciprocal behaviors.
    Inputs are in [0.5, 2], so reciprocals are safe and well-conditioned.
    
    Basis:
    0: 1
    1: x
    2: y
    3: z
    4: xy
    5: xz
    6: yz
    7: 1/x
    8: 1/y
    9: 1/z
    """
    f0 = np.ones_like(x)
    f1 = x
    f2 = y
    f3 = z
    f4 = x * y
    f5 = x * z
    f6 = y * z
    f7 = 1.0 / x
    f8 = 1.0 / y
    f9 = 1.0 / z
    
    return (params[0] * f0 +
            params[1] * f1 +
            params[2] * f2 +
            params[3] * f3 +
            params[4] * f4 +
            params[5] * f5 +
            params[6] * f6 +
            params[7] * f7 +
            params[8] * f8 +
            params[9] * f9)
# EVOLVE-BLOCK-END
