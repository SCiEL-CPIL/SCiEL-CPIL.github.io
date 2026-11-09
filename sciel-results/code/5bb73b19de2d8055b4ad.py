"""Symbolic regression with well-conditioned smooth basis: Chebyshev-like poly, multi-freq sinusoids, Gaussian bump, tanh."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible model with well-conditioned basis functions.
    
    params (length 10):
        p0: coefficient for constant
        p1: coefficient for T1(x) ~ x
        p2: coefficient for T2(x) ~ 2x^2-1
        p3: coefficient for T3(x) ~ 4x^3-3x
        p4: coefficient for T4(x) ~ 8x^4-8x^2+1
        p5: amplitude for sin(2.0*x)
        p6: amplitude for cos(2.0*x)
        p7: amplitude for sin(4.0*x)
        p8: amplitude for Gaussian bump centered at 0
        p9: coefficient for tanh(x)
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Scaled Chebyshev-like polynomials on [-2,2] mapped to [-1,1]
    # t = x/2 maps [-2,2] to [-1,1]
    t = x * 0.5
    
    # Chebyshev polynomials of first kind (well-conditioned on [-1,1])
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * t - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t * t + 1.0
    
    # Multi-frequency sinusoids (fixed frequencies avoid non-convex frequency optimization)
    sin_2x = np.sin(2.0 * x)
    cos_2x = np.cos(2.0 * x)
    sin_4x = np.sin(4.0 * x)
    
    # Gaussian bump centered at 0 with fixed width (captures local features)
    gauss_bump = np.exp(-x * x)
    
    # Tanh for smooth saturation
    tanh_term = np.tanh(x)
    
    # Combine all terms
    result = (
        params[0] * T0
        + params[1] * T1
        + params[2] * T2
        + params[3] * T3
        + params[4] * T4
        + params[5] * sin_2x
        + params[6] * cos_2x
        + params[7] * sin_4x
        + params[8] * gauss_bump
        + params[9] * tanh_term
    )
    
    # Ensure finite output
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
