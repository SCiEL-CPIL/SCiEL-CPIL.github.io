"""Symbolic regression with 1 input(s) using RBF basis with adaptive centers."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the unknown function using a linear combination of Radial Basis Functions (RBFs).
    This provides local flexibility to capture sharp features while maintaining smoothness.
    """
    x = np.asarray(x, dtype=np.float64)
    
    if not np.all(np.isfinite(x)):
        return np.zeros_like(x)
        
    n_points = x.shape[0]
    if n_points == 0:
        return np.array([])
        
    # Define 9 RBF centers spread across the domain [0.1, 6.0]
    # Plus 1 constant term for bias
    centers = np.linspace(0.1, 6.0, 9)
    
    # Width parameter for RBFs (Gaussian)
    # A smaller width means more localized functions
    # We use a moderate width to balance locality and smoothness
    width = 0.8
    
    # Build the RBF basis matrix
    # Shape: (n_points, 10)
    basis = np.zeros((n_points, 10), dtype=np.float64)
    
    # Column 0: constant term (bias)
    basis[:, 0] = 1.0
    
    # Columns 1-9: Gaussian RBFs
    for i, c in enumerate(centers):
        # Gaussian: exp(-width^2 * (x - c)^2)
        basis[:, i + 1] = np.exp(-width ** 2 * (x - c) ** 2)
        
    # Linear combination
    # params is shape (10,)
    pred = basis @ params
    
    return pred
# EVOLVE-BLOCK-END
