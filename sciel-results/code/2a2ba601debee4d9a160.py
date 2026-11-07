"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Structural change: Use a Fourier basis instead of RBF.
    # This is a global basis, good for smooth functions.
    # Map x from [-2, 2] to [-pi, pi] for standard Fourier frequencies.
    u = x * (np.pi / 2.0)
    
    # We have 10 parameters.
    # Let's use a constant term, 4 full harmonics (cos + sin), and one extra cosine.
    # Basis:
    # p0: 1
    # p1: cos(u)
    # p2: sin(u)
    # p3: cos(2u)
    # p4: sin(2u)
    # p5: cos(3u)
    # p6: sin(3u)
    # p7: cos(4u)
    # p8: sin(4u)
    # p9: cos(5u)
    
    basis = np.zeros((len(x), 10))
    basis[:, 0] = 1.0
    basis[:, 1] = np.cos(u)
    basis[:, 2] = np.sin(u)
    basis[:, 3] = np.cos(2 * u)
    basis[:, 4] = np.sin(2 * u)
    basis[:, 5] = np.cos(3 * u)
    basis[:, 6] = np.sin(3 * u)
    basis[:, 7] = np.cos(4 * u)
    basis[:, 8] = np.sin(4 * u)
    basis[:, 9] = np.cos(5 * u)
    
    return basis @ params
# EVOLVE-BLOCK-END
