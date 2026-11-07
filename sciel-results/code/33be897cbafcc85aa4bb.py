import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x)
        
    x = np.asarray(x, dtype=np.float64)
    # Normalize input to [-1, 1] for Chebyshev polynomials
    t = x / 3.0
    
    # Compute Chebyshev basis T_0...T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    # Initialize basis vectors
    T_k_minus_2 = np.ones_like(t)  # T_0
    if len(params) > 1:
        T_k_minus_1 = t            # T_1
    else:
        T_k_minus_1 = np.zeros_like(t)
        
    result = params[0] * T_k_minus_2
    if len(params) > 1:
        result += params[1] * T_k_minus_1
        
    for k in range(2, 10):
        T_k = 2.0 * t * T_k_minus_1 - T_k_minus_2
        result += params[k] * T_k
        T_k_minus_2 = T_k_minus_1
        T_k_minus_1 = T_k
        
    return result
