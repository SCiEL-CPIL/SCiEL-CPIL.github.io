import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x from [-2, 2] to [-1, 1] for Chebyshev stability
    u = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(u) = 1
    # T_1(u) = u
    # T_{k+1}(u) = 2*u*T_k(u) - T_{k-1}(u)
    
    T = np.zeros((10, len(x)), dtype=np.float64)
    T[0, :] = 1.0
    T[1, :] = u
    
    for k in range(2, 10):
        T[k, :] = 2.0 * u * T[k-1, :] - T[k-2, :]
        
    # Linear combination
    # params shape (10,), T shape (10, N)
    # result shape (N,)
    return np.dot(T, params)
