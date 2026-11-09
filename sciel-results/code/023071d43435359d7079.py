# EVOLVE-BLOCK-START
import numpy as np
import time


def _chebyshev_poly(n: int, t: np.ndarray) -> np.ndarray:
    """Compute Chebyshev polynomial T_n(t) for array t."""
    if n == 0:
        return np.ones_like(t)
    if n == 1:
        return t.copy()
    Tn_minus_1 = np.ones_like(t)
    Tn = t.copy()
    for _ in range(2, n + 1):
        Tn_plus_1 = 2.0 * t * Tn - Tn_minus_1
        Tn_minus_1 = Tn
        Tn = Tn_plus_1
    return Tn


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Chebyshev polynomial basis.
    
    We model the acceleration as a linear combination of Chebyshev polynomials
    of the normalized position and velocity. This provides good conditioning
    and allows BFGS to converge reliably.
    
    Normalization: x is in [-0.5728, 0.5275], v is in [-0.1754, 0.2821].
    We map these to [-1, 1] for Chebyshev evaluation.
    
    params (length 10):
      p[0]: coefficient for T_0(x) * T_0(v)
      p[1]: coefficient for T_1(x) * T_0(v)
      p[2]: coefficient for T_2(x) * T_0(v)
      p[3]: coefficient for T_3(x) * T_0(v)
      p[4]: coefficient for T_0(x) * T_1(v)
      p[5]: coefficient for T_1(x) * T_1(v)
      p[6]: coefficient for T_2(x) * T_1(v)
      p[7]: coefficient for T_0(x) * T_2(v)
      p[8]: coefficient for T_1(x) * T_2(v)
      p[9]: coefficient for T_2(x) * T_2(v)
    """
    # Normalize x and v to [-1, 1]
    x_min, x_max = -0.5728, 0.5275
    v_min, v_max = -0.1754, 0.2821
    
    x_norm = 2.0 * (x - (x_min + x_max) / 2.0) / (x_max - x_min)
    v_norm = 2.0 * (v - (v_min + v_max) / 2.0) / (v_max - v_min)
    
    # Clip to [-1, 1] for safety (training range should be within bounds)
    x_norm = np.clip(x_norm, -1.0, 1.0)
    v_norm = np.clip(v_norm, -1.0, 1.0)
    
    # Compute Chebyshev polynomials up to order 3 for x, order 2 for v
    Tx0 = np.ones_like(x_norm)
    Tx1 = x_norm
    Tx2 = 2.0 * x_norm * x_norm - 1.0
    Tx3 = 4.0 * x_norm**3 - 3.0 * x_norm
    
    Tv0 = np.ones_like(v_norm)
    Tv1 = v_norm
    Tv2 = 2.0 * v_norm * v_norm - 1.0
    
    # Build the prediction as a sum of products
    result = (
        params[0] * Tx0 * Tv0 +
        params[1] * Tx1 * Tv0 +
        params[2] * Tx2 * Tv0 +
        params[3] * Tx3 * Tv0 +
        params[4] * Tx0 * Tv1 +
        params[5] * Tx1 * Tv1 +
        params[6] * Tx2 * Tv1 +
        params[7] * Tx0 * Tv2 +
        params[8] * Tx1 * Tv2 +
        params[9] * Tx2 * Tv2
    )
    
    return result
# EVOLVE-BLOCK-END
