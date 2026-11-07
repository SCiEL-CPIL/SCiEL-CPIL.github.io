"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Use a 10-term Fourier basis (sines and cosines with frequencies 1-5)
    on the domain [-3, 3]. This is a strictly linear-in-params model that
    is well-conditioned and can capture both polynomial-like and periodic
    components of the target function.
    
    The basis functions are:
        cos(x), sin(x), cos(2x), sin(2x), cos(3x), sin(3x),
        cos(4x), sin(4x), cos(5x), sin(5x)
    
    This gives 10 basis functions matching the 10 free parameters.
    """
    x = np.asarray(x, dtype=float)
    
    # Build the Fourier basis matrix
    # Columns: cos(x), sin(x), cos(2x), sin(2x), ..., cos(5x), sin(5x)
    n_points = len(x)
    basis = np.empty((n_points, 10), dtype=float)
    
    for k in range(1, 6):
        basis[:, 2 * (k - 1)] = np.cos(k * x)
        basis[:, 2 * (k - 1) + 1] = np.sin(k * x)
    
    return basis @ params
# EVOLVE-BLOCK-END
