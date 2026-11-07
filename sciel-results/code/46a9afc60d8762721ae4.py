import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Check deadline
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x)
    
    # Map to [-1, 1]
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0
    T = [np.ones_like(t)]
    # T_1
    T.append(t)
    
    # T_k for k=2..9
    for k in range(2, 10):
        T_k = 2.0 * t * T[k-1] - T[k-2]
        T.append(T_k)
        
    # Combine with params
    # params is length 10
    y = np.zeros_like(x)
    for i in range(10):
        y += params[i] * T[i]
        
    return y
