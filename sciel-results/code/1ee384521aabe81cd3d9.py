"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fourier series approximation with fixed orthogonal frequencies.
    
    Basis:
    0: 1
    1: sin(pi/2 * x)
    2: cos(pi/2 * x)
    3: sin(pi * x)
    4: cos(pi * x)
    5: sin(3*pi/2 * x)
    6: cos(3*pi/2 * x)
    7: sin(2*pi * x)
    8: cos(2*pi * x)
    9: sin(5*pi/2 * x)
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    
    # Build the basis matrix A of shape (n, 10)
    # This approach is more robust and potentially faster due to BLAS usage
    basis = np.empty((n, 10), dtype=float)
    
    # Constant term
    basis[:, 0] = 1.0
    
    # Sine and Cosine terms for k = 1 to 4
    # k=1: freq = pi/2
    # k=2: freq = pi
    # k=3: freq = 3*pi/2
    # k=4: freq = 2*pi
    
    # We can vectorize the frequency calculation
    # But since we have specific slots, let's fill them directly
    
    # k=1
    basis[:, 1] = np.sin(0.5 * np.pi * x)
    basis[:, 2] = np.cos(0.5 * np.pi * x)
    
    # k=2
    basis[:, 3] = np.sin(1.0 * np.pi * x)
    basis[:, 4] = np.cos(1.0 * np.pi * x)
    
    # k=3
    basis[:, 5] = np.sin(1.5 * np.pi * x)
    basis[:, 6] = np.cos(1.5 * np.pi * x)
    
    # k=4
    basis[:, 7] = np.sin(2.0 * np.pi * x)
    basis[:, 8] = np.cos(2.0 * np.pi * x)
    
    # k=5 (Sine only, as per previous design constraint of 10 params)
    basis[:, 9] = np.sin(2.5 * np.pi * x)
    
    # Compute prediction
    return basis @ params
# EVOLVE-BLOCK-END
