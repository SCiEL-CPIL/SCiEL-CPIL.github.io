"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def _chebyshev(t, n):
    """Compute Chebyshev polynomial T_n(t) using recurrence."""
    if n == 0:
        return np.ones_like(t)
    if n == 1:
        return t.copy()
    T0 = np.ones_like(t)
    T1 = t.copy()
    for k in range(2, n + 1):
        T2 = 2.0 * t * T1 - T0
        T0, T1 = T1, T2
    return T1


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-2, 2] to [-1, 1] for standard Chebyshev polynomials
    t = x / 2.0
    
    # Precompute Chebyshev polynomials up to degree 6
    T = [_chebyshev(t, k) for k in range(7)]
    
    # Build 10 basis functions using products T_i * T_j
    # Chosen for linear independence and ability to capture non-polynomial features
    basis = np.empty((10, len(x)))
    basis[0] = T[0] * T[0]       # 1
    basis[1] = T[1] * T[0]       # x
    basis[2] = T[2] * T[0]       # 2x^2 - 1
    basis[3] = T[3] * T[0]       # 4x^3 - 3x
    basis[4] = T[4] * T[0]       # 8x^4 - 8x^2 + 1
    basis[5] = T[5] * T[0]       # 16x^5 - 20x^3 + 5x
    basis[6] = T[6] * T[0]       # 32x^6 - 48x^4 + 18x^2 - 1
    basis[7] = T[1] * T[1]       # x^2
    basis[8] = T[2] * T[1]       # 2x^3 - x
    basis[9] = T[3] * T[1]       # 4x^4 - 3x^2
    
    # Linear combination: params @ basis
    return params @ basis
# EVOLVE-BLOCK-END
