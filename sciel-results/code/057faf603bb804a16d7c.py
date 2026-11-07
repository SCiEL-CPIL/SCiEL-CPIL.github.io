"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a Chebyshev polynomial expansion.
    
    The input x is normalized to the range [-1, 1] to improve numerical stability.
    The model uses the first 10 Chebyshev polynomials of the first kind (T0 to T9)
    as a basis. The 10 parameters correspond to the coefficients of these basis functions.
    
    Chebyshev polynomials are well-conditioned for approximation on [-1, 1] and
    allow the BFGS optimizer to efficiently find the best linear combination
    that minimizes the training MSE.
    """
    # Deadline guard: The evaluator kills after 150s.
    # This function is O(N) and extremely fast, but we include the guard as a safety measure.
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill
    
    # Ensure x is a 1D array for broadcasting
    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    
    # Normalize input to [-1, 1] from [-3, 3]
    # This keeps the values of the basis functions bounded and helps with gradient scaling
    u = x / 3.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(u) = 1
    # T_1(u) = u
    # T_k(u) = 2*u*T_{k-1}(u) - T_{k-2}(u)
    
    # Initialize the result with the first term
    # We will accumulate the weighted sum
    T_prev2 = np.ones(n)  # T_0
    result = params[0] * T_prev2
    
    if len(params) > 1:
        T_prev1 = u.copy() # T_1
        result += params[1] * T_prev1
        
        # Compute T_k for k=2 to 9
        for k in range(2, 10):
            if k < len(params):
                T_curr = 2.0 * u * T_prev1 - T_prev2
                result += params[k] * T_curr
                T_prev2 = T_prev1
                T_prev1 = T_curr
            else:
                # If params is shorter than expected, stop early (though task says length 10)
                break
        
        # Safety check: if computation took too long (unlikely), return what we have.
        if time.time() > deadline:
             return result

    return result
# EVOLVE-BLOCK-END
