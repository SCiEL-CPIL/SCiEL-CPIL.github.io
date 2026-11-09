import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x from [-3, 3] to [-1, 1] for Chebyshev polynomials
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 through T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    N = 10
    T = np.zeros((len(x), N))
    T[:, 0] = 1.0
    T[:, 1] = t
    for k in range(2, N):
        T[:, k] = 2.0 * t * T[:, k-1] - T[:, k-2]
    
    # Linear combination
    return T @ params
