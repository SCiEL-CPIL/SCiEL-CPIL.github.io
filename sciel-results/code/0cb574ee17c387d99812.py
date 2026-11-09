import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Chebyshev polynomial basis of degree 9.
    
    Parameters:
    x (np.ndarray): Input array of shape (N,), values in [-3, 3].
    params (np.ndarray): Array of 10 coefficients to be fitted.
    
    Returns:
    np.ndarray: Predictions of shape (N,).
    """
    # Deadline guard (strictly not needed for this speed, but required by directive)
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin from 120s limit
    
    # Map x from [-3, 3] to [-1, 1] for Chebyshev polynomials
    t = x / 3.0
    
    # Precompute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2*t*T_n(t) - T_{n-1}(t)
    
    # Initialize basis array
    # We will compute each T_k and add to the result
    result = np.zeros_like(t)
    
    # T_0
    T_nm1 = np.ones_like(t)
    result += params[0] * T_nm1
    
    if len(t) == 0:
        return result
        
    # T_1
    T_n = t
    result += params[1] * T_n
    
    # Check deadline
    if time.time() > deadline:
        return result
        
    # Compute T_2 to T_9
    for k in range(2, 10):
        T_np1 = 2.0 * t * T_n - T_nm1
        result += params[k] * T_np1
        T_nm1 = T_n
        T_n = T_np1
        
        # Check deadline periodically
        if time.time() > deadline:
            break
            
    return result
