import numpy as np
import time

# Precompute Chebyshev basis values?
# We can't precompute for x because x changes.
# But we can precompute the coefficients for T_k(t) in terms of t?
# T_k(t) is a polynomial in t.
# T_0 = 1
# T_1 = t
# T_2 = 2t^2 - 1
# T_3 = 4t^3 - 3t
# ...
# We can compute these recursively or use the closed form.
# Recursion is fast: T_{k+1}(t) = 2t T_k(t) - T_{k-1}(t)

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    # The evaluator kills after 150s.
    # This function is called by BFGS.
    # We should be fast.
    
    # Map x to [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 ... T_9
    # Shape: (N, 10)
    N = len(x)
    basis = np.zeros((N, 10))
    
    # T_0
    basis[:, 0] = 1.0
    # T_1
    basis[:, 1] = t
    
    # T_k for k >= 2
    for k in range(2, 10):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
        
    # Linear combination
    return basis @ params
