"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Universal approximator using 5 Gaussian Radial Basis Functions with
    learnable centers, widths, and amplitudes (10 parameters total).
    
    Parameters:
    - p0, p1, p2, p3, p4: centers of the 5 Gaussians
    - p5, p6, p7, p8, p9: amplitudes of the 5 Gaussians
    
    Widths are fixed at 1.0 for stability during BFGS optimization.
    This non-linear, adaptive architecture can localize to match the
    specific structure of the hidden target function.
    """
    x = np.asarray(x)
    
    # Extract parameters: 5 centers and 5 amplitudes
    centers = params[:5]
    amplitudes = params[5:]
    
    # Fixed width for stability
    width = 1.0
    
    # Compute sum of Gaussian RBFs
    result = np.zeros_like(x, dtype=float)
    for i in range(5):
        result += amplitudes[i] * np.exp(-((x - centers[i]) ** 2) / (2.0 * width ** 2))
    
    return result
# EVOLVE-BLOCK-END
