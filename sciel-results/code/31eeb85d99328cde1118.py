import numpy as np

def _chebyshev_basis(x):
    """
    Compute Chebyshev polynomials T_0 to T_9 for input x.
    x is assumed to be in [-1, 1].
    Returns an array of shape (N, 10).
    """
    N = x.shape[0]
    basis = np.empty((N, 10))
    if N == 0:
        return basis
    basis[:, 0] = 1.0
    basis[:, 1] = x
    for k in range(2, 10):
        basis[:, k] = 2.0 * x * basis[:, k-1] - basis[:, k-2]
    return basis

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-2, 2] to [-1, 1] to utilize Chebyshev stability
    t = x * 0.5
    # Compute the 10-dimensional Chebyshev basis
    basis = _chebyshev_basis(t)
    # Linear combination of basis functions
    return basis @ params
