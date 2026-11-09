"""Symbolic regression candidate using Gaussian RBFs with learnable centers and widths."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Universal approximator using 5 Gaussian RBFs.
    
    Parameters (10 total):
    - params[0:5]: amplitudes a_i
    - params[5:10]: centers c_i
    
    Widths are fixed to avoid degenerate solutions during BFGS optimization.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Extract parameters
    a = params[0:5]   # amplitudes
    c = params[5:10]  # centers
    
    # Fixed widths to keep optimization stable
    # Spread widths to cover different scales
    widths = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
    
    # Compute RBF contributions
    # x shape: (N,), c shape: (5,)
    # diff shape: (N, 5)
    diff = x[:, np.newaxis] - c[np.newaxis, :]
    
    # Gaussian RBF: exp(-diff^2 / (2 * width^2))
    # widths shape: (5,)
    w2 = 2.0 * widths[np.newaxis, :] ** 2
    rbf = np.exp(-diff ** 2 / w2)
    
    # Weighted sum: (N, 5) @ (5,) -> (N,)
    y = rbf @ a
    
    return y
# EVOLVE-BLOCK-END
