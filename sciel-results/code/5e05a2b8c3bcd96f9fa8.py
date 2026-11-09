"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Use a rich basis function expansion with 10 parameters.
    # Basis: 1, x, x^2, x^3, x^4, x^5, x^6, x^7, x^8, x^9
    # This allows fitting polynomials up to degree 9.
    # The evaluator fits params using BFGS starting from all ones.
    # We construct the design matrix implicitly for efficiency.
    
    # Ensure x is a 1D array
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Build polynomial basis: x^0, x^1, ..., x^9
    # Use np.vander for efficient computation
    # np.vander(x, N=10, increasing=True) gives columns [1, x, x^2, ..., x^9]
    basis = np.vander(x, N=10, increasing=True)
    
    # Prediction is basis @ params
    return basis @ params
# EVOLVE-BLOCK-END
