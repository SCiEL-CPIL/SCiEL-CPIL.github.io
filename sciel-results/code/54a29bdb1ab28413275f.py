"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Chebyshev polynomials T_0..T_5 (global smooth trend) +
    Fourier terms sin/cos at frequencies 1, 2 (oscillations).
    
    The 10 parameters weight:
      0-5: Chebyshev T_0 to T_5 on normalized x in [-1, 1]
      6-9: sin(x), cos(x), sin(2x), cos(2x)
    
    This is linear in parameters, so BFGS finds the global least-squares optimum.
    """
    start_time = time.time()
    
    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    
    if n == 0:
        return np.array([])
    
    # Normalize to [-1, 1] for Chebyshev stability (x is in [-3, 3])
    t = x / 3.0
    
    # Chebyshev polynomials T_0 to T_5 using recurrence T_k = 2t*T_{k-1} - T_{k-2}
    T0 = np.ones(n)
    T1 = t.copy()
    T2 = 2.0 * t * T1 - T0
    T3 = 2.0 * t * T2 - T1
    T4 = 2.0 * t * T3 - T2
    T5 = 2.0 * t * T4 - T3
    
    # Fourier terms for low frequencies
    S1 = np.sin(x)
    C1 = np.cos(x)
    S2 = np.sin(2.0 * x)
    C2 = np.cos(2.0 * x)
    
    # Stack basis functions
    # Order: T0, T1, T2, T3, T4, T5, S1, C1, S2, C2
    basis = np.vstack([T0, T1, T2, T3, T4, T5, S1, C1, S2, C2])
    
    # Compute linear combination
    result = basis.T @ params
    
    # Safety: ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
    
    # Deadline guard (should be well within limits for this linear model)
    elapsed = time.time() - start_time
    if elapsed > 140.0:
        # If somehow slow, return what we have (unlikely)
        pass
    
    return result
# EVOLVE-BLOCK-END
