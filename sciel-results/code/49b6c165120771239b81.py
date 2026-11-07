"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    if n == 0:
        return np.array([])

    # Radial Basis Functions (RBFs) with Gaussian kernels
    # 10 centers spread across [-3, 3] with varying widths
    # This allows local adaptation to non-linear features
    
    # Centers: spread across the domain
    centers = np.array([
        -2.7, -1.8, -0.9, 0.0, 0.9, 1.8, 2.7, -2.25, -0.45, 2.25
    ])
    
    # Widths: varying to capture different scales of variation
    # Mix of narrow (local) and wide (global) RBFs
    widths = np.array([
        0.3, 0.5, 0.7, 1.0, 0.7, 0.5, 0.3, 0.8, 0.6, 0.8
    ])
    
    # Compute RBF values: phi_i(x) = exp(-((x - c_i)^2) / (2 * sigma_i^2))
    # params[i] scales each RBF independently
    
    result = np.zeros(n, dtype=np.float64)
    
    for i in range(10):
        diff = x - centers[i]
        # Gaussian RBF
        rbf_val = np.exp(-0.5 * (diff / widths[i]) ** 2)
        result += params[i] * rbf_val
    
    # Check deadline
    if time.time() > deadline:
        pass
        
    return result
# EVOLVE-BLOCK-END
