"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Uses a trigonometric basis to capture periodic/oscillatory behavior.
    The basis consists of products of sine and cosine functions with various
    frequencies. This allows the model to represent functions that are not
    well-approximated by polynomials, such as those with oscillatory components.
    
    Basis functions:
    0: 1
    1: cos(pi*x)
    2: cos(pi*y)
    3: sin(pi*x)
    4: sin(pi*y)
    5: cos(pi*x)*cos(pi*y)
    6: sin(pi*x)*sin(pi*y)
    7: cos(2*pi*x)
    8: cos(2*pi*y)
    9: sin(2*pi*x)*sin(2*pi*y)
    """
    term0 = np.ones_like(x)
    term1 = np.cos(np.pi * x)
    term2 = np.cos(np.pi * y)
    term3 = np.sin(np.pi * x)
    term4 = np.sin(np.pi * y)
    term5 = np.cos(np.pi * x) * np.cos(np.pi * y)
    term6 = np.sin(np.pi * x) * np.sin(np.pi * y)
    term7 = np.cos(2 * np.pi * x)
    term8 = np.cos(2 * np.pi * y)
    term9 = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
    
    return (params[0] * term0 +
            params[1] * term1 +
            params[2] * term2 +
            params[3] * term3 +
            params[4] * term4 +
            params[5] * term5 +
            params[6] * term6 +
            params[7] * term7 +
            params[8] * term8 +
            params[9] * term9)
# EVOLVE-BLOCK-END
