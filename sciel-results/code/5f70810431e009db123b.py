"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Mixture of 3 Tanh functions and a constant offset.
    Basis:
    y = p0 * tanh(p1 * x + p2) + p3 * tanh(p4 * x + p5) + p6 * tanh(p7 * x + p8) + p9
    This allows modeling smooth, potentially non-linear and localized features
    more robustly than a global polynomial, especially for functions with
    step-like or sigmoidal characteristics.
    """
    # Ensure x is a numpy array
    x = np.asarray(x, dtype=float)
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute the terms
    term1 = p0 * np.tanh(p1 * x + p2)
    term2 = p3 * np.tanh(p4 * x + p5)
    term3 = p6 * np.tanh(p7 * x + p8)
    
    result = term1 + term2 + term3 + p9
    
    return result
# EVOLVE-BLOCK-END
