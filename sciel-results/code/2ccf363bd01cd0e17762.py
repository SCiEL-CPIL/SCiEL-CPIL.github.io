import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 140.0
_MARGIN = 5.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a product-of-Chebyshev bilinear basis.
    
    The basis consists of 10 terms formed by products of low-degree Chebyshev polynomials:
    T_0*T_0, T_1*T_0, T_2*T_0, T_0*T_1, T_1*T_1, T_3*T_0, T_0*T_2, T_2*T_1, T_4*T_0, T_0*T_3
    
    This allows the model to capture sharp non-polynomial features more efficiently
    by exploiting multiplicative interactions in the Chebyshev domain.
    """
    # Deadline check
    if time.time() - _START_TIME > _DEADLINE - _MARGIN:
        # Return a simple fallback to avoid timeout
        return np.zeros_like(x, dtype=float)
    
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    # Ensure params has length 10
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
    
    # Map x from [-2, 2] to [-1, 1] for Chebyshev polynomials
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 through T_4 using recurrence
    T = np.zeros((5, len(x)))
    T[0, :] = 1.0
    T[1, :] = t
    for k in range(2, 5):
        T[k, :] = 2.0 * t * T[k-1, :] - T[k-2, :]
    
    # Build the 10 basis terms as products of Chebyshev polynomials
    # Selected pairs (i, j) for T_i * T_j:
    # (0,0), (1,0), (2,0), (0,1), (1,1), (3,0), (0,2), (2,1), (4,0), (0,3)
    pairs = [(0,0), (1,0), (2,0), (0,1), (1,1), (3,0), (0,2), (2,1), (4,0), (0,3)]
    
    N = len(x)
    basis = np.zeros((N, 10))
    
    for idx, (i, j) in enumerate(pairs):
        basis[:, idx] = T[i, :] * T[j, :]
    
    # Normalize columns to improve conditioning
    max_vals = np.max(np.abs(basis), axis=0)
    max_vals = np.where(max_vals > 1e-12, max_vals, 1.0)
    basis = basis / max_vals[np.newaxis, :]
    
    # Compute prediction
    return basis @ params
