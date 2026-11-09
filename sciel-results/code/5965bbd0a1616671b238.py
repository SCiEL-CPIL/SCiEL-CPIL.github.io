import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recover an unknown function using a Chebyshev series expansion.
    Input x is scaled to [-1, 1] for well-conditioned Chebyshev polynomials.
    10 parameters correspond to Chebyshev coefficients T_0 through T_9.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Scale input from [-3, 3] to [-1, 1]
    u = x / 3.0
    
    n_basis = 10
    n_points = x.size
    
    # Build Chebyshev basis matrix using recurrence
    # T[:, k] = T_k(u)
    T = np.zeros((n_points, n_basis), dtype=np.float64)
    T[:, 0] = 1.0  # T_0(u) = 1
    if n_basis > 1:
        T[:, 1] = u  # T_1(u) = u
    for k in range(2, n_basis):
        T[:, k] = 2.0 * u * T[:, k - 1] - T[:, k - 2]
    
    # Linear combination: y = sum_k params[k] * T_k(u)
    return T @ params
