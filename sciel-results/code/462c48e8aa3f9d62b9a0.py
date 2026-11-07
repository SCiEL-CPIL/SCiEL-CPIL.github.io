"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x)
    
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Gaussian Mixture Model with 4 Gaussians
    # Each Gaussian: amplitude, center, width
    # params layout:
    #   p[0:4]  = amplitudes
    #   p[4:8]  = centers
    #   p[8:10] = widths (shared for stability, or use p[8], p[9])
    
    # Use 2 shared widths for stability with only 10 params
    # Actually, let's use:
    #   p[0:4]  = amplitudes
    #   p[4:8]  = centers
    #   p[8]    = width for gaussians 0,1
    #   p[9]    = width for gaussians 2,3
    
    amps = p[0:4]
    centers = p[4:8]
    w0 = np.abs(p[8]) + 0.1  # ensure positive width
    w1 = np.abs(p[9]) + 0.1
    
    # Build the mixture
    result = np.zeros_like(x, dtype=np.float64)
    
    for i in range(4):
        width = w0 if i < 2 else w1
        result += amps[i] * np.exp(-0.5 * ((x - centers[i]) / width) ** 2)
    
    return result
# EVOLVE-BLOCK-END
