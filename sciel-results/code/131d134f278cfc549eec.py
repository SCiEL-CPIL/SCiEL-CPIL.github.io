"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    
    # Well-conditioned basis: Fourier features + Gaussian bumps
    # This avoids ill-conditioning from raw polynomials
    
    # Fourier features (different frequencies)
    b0 = np.ones_like(x)                          # constant
    b1 = np.sin(0.5 * x)                          # low freq sine
    b2 = np.cos(0.5 * x)                          # low freq cosine
    b3 = np.sin(1.0 * x)                          # mid freq sine
    b4 = np.cos(1.0 * x)                          # mid freq cosine
    b5 = np.sin(2.0 * x)                          # higher freq sine
    b6 = np.cos(2.0 * x)                          # higher freq cosine
    
    # Gaussian bumps at different locations
    b7 = np.exp(-0.5 * (x - 1.5) ** 2)            # bump at x=1.5
    b8 = np.exp(-0.5 * (x + 1.5) ** 2)            # bump at x=-1.5
    b9 = np.exp(-0.25 * x ** 2)                   # broad center bump
    
    # Combine with params
    return (params[0] * b0 + params[1] * b1 + params[2] * b2 +
            params[3] * b3 + params[4] * b4 + params[5] * b5 +
            params[6] * b6 + params[7] * b7 + params[8] * b8 +
            params[9] * b9)
# EVOLVE-BLOCK-END
