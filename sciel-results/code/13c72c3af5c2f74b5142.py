"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Trigonometric harmonic basis with product and sum interactions.
    """
    p = params
    
    # Basis 0: Constant
    b0 = np.ones_like(x)
    
    # Basis 1-3: Single variable sines (low frequency)
    b1 = np.sin(x)
    b2 = np.sin(y)
    b3 = np.sin(z)
    
    # Basis 4-6: Product interactions (sin of product)
    b4 = np.sin(x * y)
    b5 = np.sin(y * z)
    b6 = np.sin(z * x)
    
    # Basis 7-9: Sum interactions (cos of sum)
    b7 = np.cos(x + y)
    b8 = np.cos(y + z)
    b9 = np.cos(z + x)
    
    return (p[0] * b0 + p[1] * b1 + p[2] * b2 + p[3] * b3 +
            p[4] * b4 + p[5] * b5 + p[6] * b6 + p[7] * b7 +
            p[8] * b8 + p[9] * b9)
# EVOLVE-BLOCK-END
