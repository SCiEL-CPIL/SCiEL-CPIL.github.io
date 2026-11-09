import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a Chebyshev polynomial basis of degree 9.
    Chebyshev polynomials are orthogonal on [-1, 1] and provide better numerical
    conditioning than monomial bases, allowing BFGS to converge more reliably
    to the optimal coefficients for smooth functions.
    """
    # Scale x from [-2, 2] to [-1, 1] for standard Chebyshev definition
    t = x / 2.0
    
    n = x.shape[0]
    # Basis matrix for T_0 to T_9
    basis = np.empty((n, 10), dtype=np.float64)
    
    # T_0(t) = 1
    basis[:, 0] = 1.0
    
    # T_1(t) = t
    basis[:, 1] = t
    
    # Recurrence relation: T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    for k in range(2, 10):
        basis[:, k] = 2.0 * t * basis[:, k - 1] - basis[:, k - 2]
        
    return basis @ params
