"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Trigonometric basis to capture periodic/non-polynomial structures
    # Scaled by pi to match domain [-1, 1] with period 2
    
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    sx = np.sin(np.pi * x)
    cx = np.cos(np.pi * x)
    sy = np.sin(np.pi * y)
    cy = np.cos(np.pi * y)
    
    return (p0 + 
            p1 * sx + p2 * cx + 
            p3 * sy + p4 * cy + 
            p5 * sx * cy + 
            p6 * cx * sy + 
            p7 * sx * sy + 
            p8 * cx * cy + 
            p9 * np.sin(np.pi * (x + y)))
# EVOLVE-BLOCK-END
