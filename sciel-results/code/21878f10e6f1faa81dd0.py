"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    t_start = time.time()
    
    n_points = len(x)
    if n_points == 0:
        return np.array([])
        
    # Map x from [-3, 3] to t in [-1, 1] for better numerical conditioning
    t = x / 3.0
    
    # Strategy: Hybrid basis of low-degree monomials and low-frequency Fourier terms.
    # This approach aims to capture both polynomial and oscillatory components of the target function.
    # The basis is designed to be well-conditioned and smooth, which helps BFGS converge from the all-ones initialization.
    #
    # Basis functions:
    # 0: 1 (constant)
    # 1: t (linear)
    # 2: t^2 (quadratic)
    # 3: t^3 (cubic)
    # 4: t^4 (quartic)
    # 5: t^5 (quintic)
    # 6: cos(2*pi*t) (low-frequency cosine)
    # 7: sin(2*pi*t) (low-frequency sine)
    # 8: cos(4*pi*t) (higher-frequency cosine)
    # 9: sin(4*pi*t) (higher-frequency sine)
    #
    # This basis is more flexible than a pure linear model but less complex than high-degree polynomials,
    # which were found to be ill-conditioned and difficult for BFGS to optimize.
    
    basis = np.zeros((10, n_points))
    basis[0, :] = 1.0
    basis[1, :] = t
    basis[2, :] = t * t
    basis[3, :] = t * t * t
    basis[4, :] = t**4
    basis[5, :] = t**5
    basis[6, :] = np.cos(2.0 * np.pi * t)
    basis[7, :] = np.sin(2.0 * np.pi * t)
    basis[8, :] = np.cos(4.0 * np.pi * t)
    basis[9, :] = np.sin(4.0 * np.pi * t)
    
    if time.time() - t_start > 100:
        return params[0] * t + params[1]
        
    return np.einsum('i,ij->j', params, basis)
# EVOLVE-BLOCK-END
