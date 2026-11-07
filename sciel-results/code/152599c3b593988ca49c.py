"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Mixed basis: Polynomial (degree 2) + Exponential terms
    # This subspace is more flexible than pure polynomials and can approximate
    # functions with exponential character which cubic polynomials fit poorly.
    
    term0 = np.ones_like(x)
    term1 = x
    term2 = y
    term3 = x * x
    term4 = y * y
    term5 = x * y
    
    # Exponential terms to capture non-polynomial smooth behavior
    term6 = np.exp(x)
    term7 = np.exp(y)
    term8 = np.exp(x + y)
    term9 = np.exp(x - y)
    
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
