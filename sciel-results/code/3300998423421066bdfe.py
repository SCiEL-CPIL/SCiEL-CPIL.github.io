"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter basis mixing orthogonal Chebyshev polynomials and nonlinear terms.
    Uses Chebyshev polynomials of degree 0-4 in x and y for better conditioning,
    plus tanh for capturing sharp nonlinearities.
    
    Basis: T0(x), T1(x), T2(x), T3(x), T4(x), T0(y), T1(y), T2(y), tanh(x), tanh(y)
    
    Chebyshev polynomials:
    T0(t) = 1
    T1(t) = t
    T2(t) = 2t^2 - 1
    T3(t) = 4t^3 - 3t
    T4(t) = 8t^4 - 8t^2 + 1
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute Chebyshev polynomials in x
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    Tx0 = np.ones_like(x)
    Tx1 = x
    Tx2 = 2 * x2 - 1
    Tx3 = 4 * x3 - 3 * x
    Tx4 = 8 * x4 - 8 * x2 + 1
    
    # Precompute Chebyshev polynomials in y
    y2 = y * y
    y3 = y2 * y
    y4 = y2 * y2
    Ty0 = np.ones_like(y)
    Ty1 = y
    Ty2 = 2 * y2 - 1
    Ty3 = 4 * y3 - 3 * y
    Ty4 = 8 * y4 - 8 * y2 + 1
    
    # Nonlinear terms
    tanh_x = np.tanh(x)
    tanh_y = np.tanh(y)
    
    return (p0 * Tx0 + p1 * Tx1 + p2 * Tx2 + p3 * Tx3 + p4 * Tx4 +
            p5 * Ty0 + p6 * Ty1 + p7 * Ty2 + p8 * tanh_x + p9 * tanh_y)
# EVOLVE-BLOCK-END
