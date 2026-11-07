import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial basis of degree 9 with 10 parameters.
    Maps x from [-3, 3] to [-1, 1] to use standard Chebyshev polynomials T_n(t).
    The 10 params are the coefficients for T_0 to T_9.
    """
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    
    if n == 0:
        return np.array([])
    
    # Map x in [-3, 3] to t in [-1, 1]
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    T = np.zeros((10, n))
    T[0, :] = 1.0
    T[1, :] = t
    
    for i in range(2, 10):
        T[i, :] = 2.0 * t * T[i-1, :] - T[i-2, :]
        
    # Compute linear combination: y = sum(params[i] * T[i](x))
    # T is (10, n), params is (10,)
    # We want result of shape (n,)
    result = T.T @ params
    
    # Check deadline
    if time.time() > deadline:
        return result
        
    return result
