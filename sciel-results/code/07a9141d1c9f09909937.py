import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Orthonormal Legendre polynomial basis for symbolic regression.
    
    params: array of length 10, coefficients for the orthonormal Legendre basis.
    x: array of input points in [-3, 3].
    
    We map x from [-3, 3] to [-1, 1] and evaluate the orthonormal Legendre
    polynomials there. This gives a well-conditioned basis for BFGS optimization.
    """
    x = np.asarray(x, dtype=np.float64)
    t = x / 3.0  # Map to [-1, 1]
    
    # Evaluate orthonormal Legendre polynomials P_0 through P_9 on t
    # Using the three-term recurrence for standard Legendre polynomials,
    # then normalizing by sqrt((2n+1)/2) for orthonormality on [-1,1].
    
    n = 10
    # Build the design matrix: rows are points, columns are basis functions
    # Legendre recurrence: (n+1)P_{n+1}(t) = (2n+1)t P_n(t) - n P_{n-1}(t)
    P = np.zeros((len(t), n), dtype=np.float64)
    P[:, 0] = 1.0  # P_0(t) = 1
    if n > 1:
        P[:, 1] = t  # P_1(t) = t
    for k in range(2, n):
        P[:, k] = ((2*k - 1) * t * P[:, k-1] - (k - 1) * P[:, k-2]) / k
    
    # Normalize each column to be orthonormal on [-1,1]
    # ||P_k||^2 = 2/(2k+1), so normalized P_k = P_k * sqrt((2k+1)/2)
    for k in range(n):
        P[:, k] *= np.sqrt((2*k + 1) / 2.0)
    
    # Return P @ params
    return P @ params
