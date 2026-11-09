"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    Uses a 2D Chebyshev polynomial basis mapped to [-1, 1].
    This basis is well-conditioned for least squares fitting.
    """
    # Input ranges from task description
    x_min, x_max = -0.5728, 0.5275
    v_min, v_max = -0.1754, 0.2821
    
    # Map inputs to [-1, 1]
    # Handle division by zero just in case, though ranges are fixed
    x_range = x_max - x_min
    v_range = v_max - v_min
    
    if x_range == 0:
        u = np.zeros_like(x)
    else:
        u = 2.0 * (x - (x_max + x_min) / 2.0) / x_range
        
    if v_range == 0:
        w = np.zeros_like(v)
    else:
        w = 2.0 * (v - (v_max + v_min) / 2.0) / v_range


    # Compute Legendre polynomials L0 to L3
    # L0 = 1
    # L1 = x
    # L2 = 0.5 * (3*x^2 - 1)
    # L3 = 0.5 * (5*x^3 - 3*x)
    L0_u = np.ones_like(u)
    L1_u = u
    L2_u = 0.5 * (3.0 * u**2 - 1.0)
    L3_u = 0.5 * (5.0 * u**3 - 3.0 * u)
    
    L0_w = np.ones_like(w)
    L1_w = w
    L2_w = 0.5 * (3.0 * w**2 - 1.0)
    L3_w = 0.5 * (5.0 * w**3 - 3.0 * w)
    
    # Construct 10 basis terms using Legendre polynomials
    # p0: L0(u) L0(w)
    # p1: L1(u) L0(w)
    # p2: L0(u) L1(w)
    # p3: L2(u) L0(w)
    # p4: L0(u) L2(w)
    # p5: L1(u) L1(w)
    # p6: L3(u) L0(w)
    # p7: L0(u) L3(w)
    # p8: L2(u) L1(w)
    # p9: L1(u) L2(w)
    
    term0 = L0_u * L0_w
    term1 = L1_u * L0_w
    term2 = L0_u * L1_w
    term3 = L2_u * L0_w
    term4 = L0_u * L2_w
    term5 = L1_u * L1_w
    term6 = L3_u * L0_w
    term7 = L0_u * L3_w
    term8 = L2_u * L1_w
    term9 = L1_u * L2_w

    return (params[0] * term0 +
            params[1] * term1 +
            params[2] * term2 +
            params[3] * term3 +
            params[4] * term4 +
            params[5] * term5 +
            params[6] * term6 +
            params[7] * term7 +
            params[8] * term8 +
            params[9] * term9)
# EVOLVE-BLOCK-END
