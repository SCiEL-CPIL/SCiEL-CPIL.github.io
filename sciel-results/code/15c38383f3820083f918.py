"""Symbolic regression with Gaussian Radial Basis Function basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Fixed centers for the 10 RBFs, evenly spaced in [-2, 2]
_CENTERS = np.linspace(-2.0, 2.0, 10)
_WIDTH = 0.5
_INV_2_WIDTH_SQ = 1.0 / (2.0 * _WIDTH ** 2)

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recover an unknown function using a fixed-center Gaussian RBF basis.
    
    The model is y(x) = sum_{i=0}^{9} params[i] * exp(-0.5 * ((x - c_i) / width)^2)
    
    This local basis allows the model to fit non-polynomial features more flexibly
    than a global polynomial of the same degree.
    """
    # Ensure x is a 1D array
    x = np.asarray(x, dtype=float)
    
    # Compute RBF values
    # x shape: (N,)
    # _CENTERS shape: (10,)
    # diff shape: (N, 10)
    diff = x[:, np.newaxis] - _CENTERS[np.newaxis, :]
    
    # Compute Gaussian values
    # rbf shape: (N, 10)
    rbf = np.exp(-0.5 * (diff / _WIDTH) ** 2)
    
    # Compute linear combination
    # params shape: (10,)
    # prediction shape: (N,)
    return rbf @ params
# EVOLVE-BLOCK-END
