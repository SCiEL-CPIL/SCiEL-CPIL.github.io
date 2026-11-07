"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    High-capacity linear basis: Chebyshev polynomials (T_0 to T_5) + Fourier terms (freq 1-5).
    
    The 10 parameters weight:
      0-5: Chebyshev T_0 to T_5 on normalized x in [-1, 1]
      6-9: sin/cos at frequencies 1, 2 (and implicitly higher via the Chebyshev tail if needed, 
           but here we explicitly add sin/cos for 1 and 2, and use the remaining params for 
           higher order Chebyshev to capture non-periodic high-order trends).
    
    Actually, to use all 10 params effectively for a general smooth function:
    - 6 params for Chebyshev T_0..T_5 (captures polynomial behavior up to degree 5)
    - 4 params for sin(x), cos(x), sin(2x), cos(2x) (captures fundamental oscillations)
    
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
    # We use sin/cos for x and 2x to cover the most common periodic components
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
        # If overflow occurred, fall back to a simpler, more stable subset or clip
        # This should rarely happen with normalized t and bounded sin/cos, but good practice
        result = np.where(np.isfinite(result), result, 0.0)
        
    # Deadline check (though this is fast, good practice per directive)
    elapsed = time.time() - start_time
    if elapsed > 140.0:
        # If we are somehow slow (unlikely for this linear model), return what we have
        pass

    return result
# EVOLVE-BLOCK-END
