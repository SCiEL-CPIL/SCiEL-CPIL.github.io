"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-term Fourier series with fixed frequencies {1, 2, 3, 4, 5}.
    Basis: cos(x), sin(x), cos(2x), sin(2x), ..., cos(5x), sin(5x).
    This is linear in params, so BFGS solves it efficiently.
    The trigonometric basis is well-conditioned on [-3, 3] and can capture
    periodic or oscillatory components that polynomials miss.
    """
    x = np.asarray(x, dtype=float)
    
    # Build the Fourier basis matrix: [cos(x), sin(x), cos(2x), sin(2x), ..., cos(5x), sin(5x)]
    n_points = len(x)
    basis = np.empty((n_points, 10), dtype=float)
    
    for k in range(1, 6):
        col_cos = 2 * (k - 1)
        col_sin = 2 * (k - 1) + 1
        basis[:, col_cos] = np.cos(k * x)
        basis[:, col_sin] = np.sin(k * x)
    
    return basis @ params
# EVOLVE-BLOCK-END
