"""Symbolic regression with a Chebyshev polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial approximation of degree 9.
    
    y = sum_{k=0}^{9} p_k * T_k(x/2)
    
    where T_k is the Chebyshev polynomial of the first kind.
    The input x is in [-2, 2], so x/2 is in [-1, 1], which is the
    standard domain for Chebyshev polynomials.
    
    params layout (length 10):
      params[0] = coefficient for T_0
      params[1] = coefficient for T_1
      ...
      params[9] = coefficient for T_9
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    p = np.asarray(params, dtype=float)
    
    # Map x from [-2, 2] to t in [-1, 1]
    t = 0.5 * x
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    T = np.ones_like(t)
    T_prev = np.ones_like(t)
    T_curr = t.copy()
    
    # We need T_0 to T_9.
    # T_0 is T_prev initially? No, let's be explicit.
    # T_0 = 1
    # T_1 = t
    # T_2 = 2t*T_1 - T_0
    # ...
    
    # Let's store them in a list or compute the sum directly.
    # To be efficient and avoid storing all, we can accumulate the result.
    
    # p[0] * T_0
    result = p[0] * T_prev
    
    # p[1] * T_1
    result += p[1] * T_curr
    
    if len(p) > 2:
        T_next = 2.0 * t * T_curr - T_prev
        result += p[2] * T_next
        
        for k in range(3, 10):
            T_prev, T_curr = T_curr, T_next
            T_next = 2.0 * t * T_curr - T_prev
            result += p[k] * T_next
            
    # Safety: guarantee finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
