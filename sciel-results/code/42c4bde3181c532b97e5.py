# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Chebyshev polynomial basis.
    
    Chebyshev polynomials are orthogonal on [-1, 1] and provide a stable
    basis for approximating smooth functions. We map the input domain
    [-2, 2] to [-1, 1] by scaling x by 0.5.
    
    Parameters (10 total):
    Coefficients for Chebyshev polynomials T_0 to T_9.
    
    The model is:
    y = sum_{k=0}^{9} p_k * T_k(x/2)
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has exactly 10 elements
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Map x from [-2, 2] to [-1, 1]
    t = 0.5 * x
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    T = np.zeros((10, x.shape[0]), dtype=np.float64)
    T[0] = 1.0
    T[1] = t
    
    for k in range(2, 10):
        T[k] = 2.0 * t * T[k-1] - T[k-2]
    
    # Linear combination of Chebyshev polynomials
    # result = sum_{k=0}^{9} p_k * T_k
    result = np.sum(p[:, np.newaxis] * T, axis=0)
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(result)):
        # If any non-finite values, return zeros or a simple fallback
        # to avoid crashing the optimizer, though evaluator handles it.
        # Returning 0 is a safe neutral value.
        result = np.where(np.isfinite(result), result, 0.0)
        
    return result
# EVOLVE-BLOCK-END
