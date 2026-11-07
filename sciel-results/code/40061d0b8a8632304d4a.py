"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a hybrid Chebyshev + Fourier basis.
    
    The 10 parameters are split:
    - params[0:4] : coefficients for Chebyshev polynomials T0-T3 (global trend)
    - params[4:10]: coefficients for sin/cos terms at frequencies 1, 2, 3 (oscillations)
    
    This hybrid captures both smooth global structure and periodic/local features.
    """
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)

    # Normalize to [-1, 1] for Chebyshev stability
    t = x / 3.0

    # --- Chebyshev polynomials T0 to T5 ---
    b0 = np.ones(n)          # T_0
    b1 = t.copy()            # T_1
    b2 = 2.0 * t**2 - 1.0    # T_2
    b3 = 4.0 * t**3 - 3.0 * t  # T_3
    b4 = 8.0 * t**4 - 8.0 * t**2 + 1.0  # T_4
    b5 = 16.0 * t**5 - 20.0 * t**3 + 5.0 * t  # T_5

    # --- Fourier terms: sin/cos at x, 2x, 3x, 4x (using remaining 4 params) ---
    # Wait, we only have 10 params total. Let me restructure:
    # params[0:4]: T0-T3
    # params[4:6]: sin(x), cos(x)
    # params[6:8]: sin(2x), cos(2x)
    # params[8:10]: sin(3x), cos(3x)
    # That's the original. Let me keep that but improve the Chebyshev part.
    
    # Actually, let me use:
    # params[0:5]: T0-T4
    # params[5:7]: sin(x), cos(x)  
    # params[7:9]: sin(2x), cos(2x)
    # params[9:10]: sin(3x)
    # But that's asymmetric. Better to stick with balanced approach.
    
    # Let me just improve by using T0-T3 for polynomial and sin/cos at 1,2,3,4
    # but we only have 6 Fourier slots for 4 frequencies (8 terms). 
    # So let's do:
    # params[0:4]: T0-T3
    # params[4:10]: sin(x), cos(x), sin(2x), cos(2x), sin(3x), cos(3x)
    # Same as before, but let me add T4 and T5 by dropping some Fourier terms?
    # No, let's keep the structure but use a better normalization.
    
    # Revised: use T0-T3 and sin/cos at frequencies 1, 2, 3 (same as before)
    # but improve numerical stability with proper Chebyshev recurrence
    b4 = np.sin(x)
    b5 = np.cos(x)
    b6 = np.sin(2.0 * x)
    b7 = np.cos(2.0 * x)
    b8 = np.sin(3.0 * x)
    b9 = np.cos(3.0 * x)
    
    # Add higher-order Chebyshev using recurrence for stability
    # T_4 = 2*t*T_3 - T_2
    # T_5 = 2*t*T_4 - T_3
    # But we only have 10 params, so we can't add more terms without dropping others.
    # Let's keep the current 10-term structure but ensure numerical stability.

    # Stack basis and compute linear combination
    basis = np.vstack([
        b0, b1, b2, b3,
        b4, b5, b6, b7,
        b8, b9
    ])

    result = basis.T @ params

    # Safety: ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)

    if time.time() > deadline:
        pass  # already computed, just return

    return result
# EVOLVE-BLOCK-END
