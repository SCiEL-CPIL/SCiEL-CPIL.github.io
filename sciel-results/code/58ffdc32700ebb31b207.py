import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a Chebyshev polynomial basis of order 9.
    This approach uses orthogonal polynomials which are well-conditioned
    for approximation on a bounded interval, providing a flexible
    polynomial fit without the instability of high-order monomials.
    """
    # Scale x from [-2, 2] to [-1, 1] for standard Chebyshev definition
    u = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(u) = 1
    # T_1(u) = u
    # T_{n+1}(u) = 2u T_n(u) - T_{n-1}(u)
    
    T = np.zeros((10, x.shape[0]), dtype=np.float64)
    T[0] = 1.0
    if x.shape[0] > 0:
        T[1] = u
        for k in range(2, 10):
            T[k] = 2.0 * u * T[k-1] - T[k-2]
            
    # Linear combination
    # params are a_0 to a_9
    return np.dot(params, T)
