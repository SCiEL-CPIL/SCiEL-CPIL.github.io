import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [-2, 2] to [-1, 1]
    t = x / 2.0
    
    N = len(x)
    # Pre-allocate for 10 Chebyshev polynomials
    C = np.zeros((10, N))
    
    # T_0(t) = 1
    C[0, :] = 1.0
    # T_1(t) = t
    C[1, :] = t
    
    # Recurrence: T_k(t) = 2t * T_{k-1}(t) - T_{k-2}(t)
    for k in range(2, 10):
        C[k, :] = 2.0 * t * C[k-1, :] - C[k-2, :]
        
    # Compute linear combination
    y = np.dot(params, C)
    
    return y
