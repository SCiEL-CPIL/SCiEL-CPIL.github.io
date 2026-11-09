"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model for a Duffing oscillator:
    a = c0 + c1*x + c2*v + c3*x^3 + c4*sin(t) + c5*cos(t)
    Uses 6 parameters. Remaining 4 are unused (set to 0 effectively by not including them).
    """
    x3 = x * x * x
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    return (params[0] + 
            params[1] * x + 
            params[2] * v + 
            params[3] * x3 + 
            params[4] * sin_t + 
            params[5] * cos_t)
# EVOLVE-BLOCK-END
