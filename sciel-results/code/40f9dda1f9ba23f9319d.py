"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Chebyshev polynomials (T0-T4) + learnable-frequency sine + learnable exponential.
    
    params[0:5] : coefficients for Chebyshev T0(z) through T4(z)
    params[5]   : frequency w for sin(w*z)
    params[6]   : amplitude A for A*sin(w*z)
    params[7]   : amplitude B for B*exp(r*z)
    params[8]   : rate r for exp(r*z)
    params[9]   : additional constant offset (redundant with T0 but helps conditioning)
    """
    z = x / 3.0  # Map [-3, 3] to [-1, 1]
    
    # --- Chebyshev polynomials T0..T4 on z in [-1, 1] ---
    T0 = np.ones_like(z)
    T1 = z
    T2 = 2.0 * z * z - 1.0
    T3 = 4.0 * z * z * z - 3.0 * z
    T4 = 8.0 * z * z * z * z - 8.0 * z * z + 1.0
    
    # --- Learnable-frequency sinusoidal term ---
    w = params[5]
    A = params[6]
    sin_term = A * np.sin(w * z)
    
    # --- Learnable exponential term ---
    r = params[8]
    B = params[7]
    # Clip r to avoid overflow: exp(r*z) with z in [-1,1], r up to ~50 would overflow
    # Use a safe clip: |r| <= 50 keeps exp(50) ~ 5e21 which is finite but large
    r_safe = np.clip(r, -50.0, 50.0)
    exp_term = B * np.exp(r_safe * z)
    
    # --- Assemble prediction ---
    pred = (params[0] * T0 +
            params[1] * T1 +
            params[2] * T2 +
            params[3] * T3 +
            params[4] * T4 +
            sin_term +
            exp_term +
            params[9] * np.ones_like(z))
    
    return pred
# EVOLVE-BLOCK-END
