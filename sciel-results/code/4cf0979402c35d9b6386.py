"""Symbolic regression: RBF basis for universal approximation."""
# EVOLVE-BLOCK-START
import numpy as np

# Pre-compute RBF centers and widths
# 10 centers uniformly spaced in [-3, 3]
_CENTERS = np.linspace(-3.0, 3.0, 10)
_WIDTH = 1.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the hidden function using a basis of 10 Gaussian Radial Basis Functions.
    
    params[0..9]: Weights for the RBFs centered at -3.0, -2.33, ..., 3.0.
    
    The evaluator will fit these 10 parameters via BFGS starting from ones.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    
    # Compute RBF values: exp(-((x - c_i)^2) / (2 * sigma^2))
    # x is (n,), _CENTERS is (10,)
    # diff is (n, 10)
    diff = x[:, np.newaxis] - _CENTERS[np.newaxis, :]
    
    # Exponent: -diff^2 / (2 * sigma^2)
    exponent = - (diff * diff) / (2.0 * _WIDTH * _WIDTH)
    
    # Basis matrix (n, 10)
    V = np.exp(exponent)
    
    return V @ params
# EVOLVE-BLOCK-END
