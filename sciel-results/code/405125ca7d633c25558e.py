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

    # Clip to [-1, 1] to stay within domain of Chebyshev definition stability
    u = np.clip(u, -1.0, 1.0)
    w = np.clip(w, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T0 to T3
    T0_u = np.ones_like(u)
    T1_u = u
    T2_u = 2.0 * u**2 - 1.0
    T3_u = 4.0 * u**3 - 3.0 * u
    
    T0_w = np.ones_like(w)
    T1_w = w
    T2_w = 2.0 * w**2 - 1.0
    T3_w = 4.0 * w**3 - 3.0 * w
    
    # Construct 10 basis terms
    # p0: T0(u) T0(w)
    # p1: T1(u) T0(w)
    # p2: T0(u) T1(w)
    # p3: T2(u) T0(w)
    # p4: T0(u) T2(w)
    # p5: T1(u) T1(w)
    # p6: T3(u) T0(w)
    # p7: T0(u) T3(w)
    # p8: T2(u) T1(w)
    # p9: T1(u) T2(w)
    
    term0 = T0_u * T0_w
    term1 = T1_u * T0_w
    term2 = T0_u * T1_w
    term3 = T2_u * T0_w
    term4 = T0_u * T2_w
    term5 = T1_u * T1_w
    term6 = T3_u * T0_w
    term7 = T0_u * T3_w
    term8 = T2_u * T1_w
    term9 = T1_u * T2_w

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
