import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the function using a Chebyshev polynomial basis of degree 9.
    The input x is assumed to be in [-2, 2]. We map it to t in [-1, 1]
    to utilize the standard Chebyshev polynomials T_k(t).
    """
    t = x / 2.0
    
    # Precompute Chebyshev basis values for all points
    # Shape: (10, N)
    basis = np.zeros((10, len(x)), dtype=float)
    basis[0, :] = 1.0
    if len(x) > 0:
        basis[1, :] = t
        for k in range(2, 10):
            basis[k, :] = 2.0 * t * basis[k-1, :] - basis[k-2, :]
            
    # Compute linear combination
    # params shape (10,), basis shape (10, N)
    return basis.T @ params
