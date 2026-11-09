"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Hybrid basis: 4 Gaussian bumps + 4 Chebyshev polynomial terms + 2 oscillatory terms.
    # This combines local flexibility (Gaussians) for potential non-smooth features
    # with global stability (Chebyshev polynomials) and bounded oscillatory components.
    
    # Gaussian basis functions centered at strategic points in [-2, 2]
    # Centers: -1.5, -0.5, 0.5, 1.5 with width ~0.8
    centers = np.array([-1.5, -0.5, 0.5, 1.5])
    width = 0.8
    
    # Compute Gaussian terms: exp(-(x - c)^2 / (2*width^2))
    x_expanded = x[:, np.newaxis]
    gaussians = np.exp(-((x_expanded - centers) ** 2) / (2 * width ** 2))
    # gaussians shape: (n_points, 4)
    
    # Chebyshev polynomial terms T_0 through T_3 on scaled domain [-1, 1]
    t = x / 2.0  # scale to [-1, 1]
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2 * t * T1 - T0
    T3 = 2 * t * T2 - T1
    
    # Oscillatory terms for additional frequency capture
    osc1 = np.sin(np.pi * x / 2.0)
    osc2 = np.cos(np.pi * x / 2.0)
    
    # Combine all basis terms: 4 gaussians + 4 Chebyshev + 2 oscillatory = 10 total
    # params[0:4] for gaussians, params[4:8] for Chebyshev, params[8:10] for oscillatory
    
    result = (gaussians[:, 0] * params[0] +
              gaussians[:, 1] * params[1] +
              gaussians[:, 2] * params[2] +
              gaussians[:, 3] * params[3] +
              T0 * params[4] +
              T1 * params[5] +
              T2 * params[6] +
              T3 * params[7] +
              osc1 * params[8] +
              osc2 * params[9])
    
    return result
# EVOLVE-BLOCK-END
