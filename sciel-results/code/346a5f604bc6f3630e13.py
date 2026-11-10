"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a sum of 5 Gaussian basis functions (RBF).
    
    Parameters:
      p[0:5]  - amplitudes of the 5 Gaussians
      p[5:10] - means (centers) of the 5 Gaussians
    
    Widths are fixed at multiple scales to capture features from sharp to broad:
      widths = [0.25, 0.5, 1.0, 1.5, 2.5]
    
    This localized basis can approximate sharp features, kinks, discontinuities,
    and multi-modal structures that global polynomial/trig bases smooth out.
    """
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Fixed widths at multiple scales
    widths = np.array([0.25, 0.5, 1.0, 1.5, 2.5])
    
    # Extract amplitudes and means
    amps = p[0:5]
    means = p[5:10]
    
    # Compute sum of 5 Gaussians: y = sum_i amps[i] * exp(-(x - means[i])^2 / (2 * widths[i]^2))
    y = np.zeros_like(x)
    for i in range(5):
        y += amps[i] * np.exp(-0.5 * ((x - means[i]) / widths[i]) ** 2)
    
    return y
# EVOLVE-BLOCK-END
