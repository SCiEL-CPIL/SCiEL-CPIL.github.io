"""Symbolic regression with pure Chebyshev polynomial basis of degree 9."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a pure Chebyshev polynomial basis of degree 9.
    
    Basis functions (10 total):
    Chebyshev polynomials T0 through T9, with input mapped from [-3, 3] to [-1, 1].
    
    Chebyshev polynomials are orthogonal on [-1, 1] with respect to the weight
    function 1/sqrt(1-t^2), providing excellent numerical conditioning for
    linear least-squares fitting via BFGS. This basis can represent any
    smooth function on the interval with high accuracy.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    n = x.shape[0]
    
    # Map input from [-3, 3] to [-1, 1]
    t = x / 3.0
    
    # Compute Chebyshev polynomials T0 through T9 using the recurrence relation:
    # T0(t) = 1
    # T1(t) = t
    # T_{k+1}(t) = 2*t*T_k(t) - T_{k-1}(t)
    T = np.zeros((n, 10))
    T[:, 0] = 1.0
    T[:, 1] = t
    for k in range(2, 10):
        T[:, k] = 2.0 * t * T[:, k-1] - T[:, k-2]
    
    # Combine all basis functions with the corresponding parameters
    result = T @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.zeros_like(x)
    
    return result
# EVOLVE-BLOCK-END
