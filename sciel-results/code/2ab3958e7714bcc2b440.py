"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure x is a numpy array for consistent operations
    x = np.asarray(x, dtype=np.float64)
    
    # Map x from [-2, 2] to t in [-1, 1] for Chebyshev stability
    t = 0.5 * x
    
    # Compute Chebyshev polynomials T_0 to T_9
    # Recurrence: T_0(t) = 1, T_1(t) = t
    # T_{n}(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    # We need 10 basis functions.
    # Instead of storing all in a matrix (which might be memory inefficient for large x, 
    # though 200 points is small), we can compute them iteratively.
    
    # Initialize the result with the first term
    # T_0 = 1
    T_prev2 = np.ones_like(t)
    
    if len(t) == 0:
        return np.array([])
        
    # T_1 = t
    T_prev1 = t
    
    result = params[0] * T_prev2 + params[1] * T_prev1
    
    for k in range(2, 10):
        # T_k = 2*t*T_{k-1} - T_{k-2}
        T_curr = 2.0 * t * T_prev1 - T_prev2
        result = result + params[k] * T_curr
        
        # Update for next iteration
        T_prev2 = T_prev1
        T_prev1 = T_curr
        
    return result
# EVOLVE-BLOCK-END
