import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Legendre polynomial series expansion.
    The input x is scaled to [-1, 1] to align with the standard Legendre domain.
    We use 10 parameters corresponding to Legendre polynomials P_0 through P_9.
    
    Legendre polynomials are orthogonal with respect to the uniform distribution
    on [-1, 1], which matches the data distribution (uniform on [-3, 3] scaled to [-1, 1]).
    This should lead to better conditioning and faster convergence for the BFGS optimizer
    compared to Chebyshev polynomials, which are orthogonal with respect to a different weight.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second deadline with safety margin
    
    x = np.asarray(x, dtype=np.float64)
    
    # Scale input to [-1, 1] for Legendre polynomials
    # Original domain is [-3, 3]
    u = x / 3.0
    
    # Number of basis functions (parameters)
    n_basis = 10
    n_points = x.size
    
    # Initialize the basis matrix
    # Shape: (n_points, n_basis)
    P = np.zeros((n_points, n_basis), dtype=np.float64)
    
    # P_0(u) = 1
    P[:, 0] = 1.0
    
    # P_1(u) = u
    P[:, 1] = u
    
    # Recurrence relation for Legendre polynomials:
    # (n+1) P_{n+1}(u) = (2n+1) u P_n(u) - n P_{n-1}(u)
    for n in range(1, n_basis - 1):
        P[:, n + 1] = ((2 * n + 1) * u * P[:, n] - n * P[:, n - 1]) / (n + 1)
    
    # Return the linear combination of basis functions
    result = P @ params
    
    # Ensure finite values
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
    
    return result
