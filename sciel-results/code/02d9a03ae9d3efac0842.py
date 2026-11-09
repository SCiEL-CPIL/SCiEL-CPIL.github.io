"""Symbolic regression with pure Chebyshev polynomial basis of degree 9."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 110.0  # Safety margin before 120s limit


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline
    if time.time() - _START_TIME > _TIME_LIMIT:
        return np.zeros_like(x)
    """
    Recovers an unknown function using a pure Chebyshev polynomial basis of degree 9.
    
    Basis functions (10 total):
    T0(x), T1(x), ..., T9(x) where Tk are Chebyshev polynomials of the first kind
    evaluated at t = x/3 (mapping [-3,3] to [-1,1]).
    
    This basis is well-conditioned, orthogonal on [-1,1] with respect to the
    weight 1/sqrt(1-t^2), and can represent any smooth function on [-3,3]
    with rapid convergence. BFGS optimization handles the linear combination
    of these basis functions efficiently.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    n = x.shape[0]
    
    # Map to [-1, 1] for Chebyshev polynomials
    t = x / 3.0
    
    # Compute Chebyshev polynomials T0 through T9 using recurrence
    T = np.zeros((10, n), dtype=np.float64)
    T[0, :] = 1.0
    if n > 0:
        T[1, :] = t
        for k in range(2, 10):
            T[k, :] = 2.0 * t * T[k-1, :] - T[k-2, :]
    
    # Combine all basis functions: result = sum(params[i] * T[i])
    # T is (10, n), params is (10,). We want result[j] = sum_i params[i] * T[i, j]
    # This is params @ T
    result = np.dot(params, T)
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.zeros_like(x)
    
    return result
# EVOLVE-BLOCK-END
