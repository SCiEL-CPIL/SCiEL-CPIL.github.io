"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale input to [-1, 1] for better numerical conditioning
    u = x / 3.0
    u2 = u * u
    u3 = u2 * u
    u4 = u2 * u2
    
    # Polynomial basis (scaled monomials for O(1) magnitude on [-1,1])
    # These capture smooth, non-periodic trends
    b0 = 1.0
    b1 = u
    b2 = u2
    b3 = u3
    b4 = u4
    
    # Trigonometric basis
    # Frequencies chosen to cover the domain [-1, 1] with fundamental period 2 (period 6 in x)
    # sin(pi*u) has period 2 in u, sin(2*pi*u) has period 1 in u
    b5 = np.sin(np.pi * u)
    b6 = np.cos(np.pi * u)
    b7 = np.sin(2.0 * np.pi * u)
    b8 = np.cos(2.0 * np.pi * u)
    
    # Gaussian bump for localized features
    # Width tuned to be significant over the domain
    b9 = np.exp(-4.0 * u2)
    
    # Combine
    return (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
# EVOLVE-BLOCK-END
