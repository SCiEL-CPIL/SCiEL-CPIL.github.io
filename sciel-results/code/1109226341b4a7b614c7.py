"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation using a linear combination of shifted Gaussian basis functions.
    
    params: array of length 10 representing amplitudes of 10 Gaussian bumps.
    The centers are fixed and evenly spaced across [-2, 2].
    The widths are fixed to provide good local coverage.
    """
    # Fixed centers evenly spaced across [-2, 2]
    centers = np.linspace(-2.0, 2.0, 10)
    
    # Fixed widths - chosen to provide good local coverage
    # Width of about 0.5 gives good overlap between adjacent centers
    width = 0.5
    
    # Compute Gaussian basis functions
    # Shape: (N, 10) where N is number of input points
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    basis = np.exp(-0.5 * (diff / width) ** 2)
    
    # Linear combination
    return basis @ params
# EVOLVE-BLOCK-END
