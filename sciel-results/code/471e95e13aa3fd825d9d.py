import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Chebyshev polynomial basis of degree 9.
    
    Strategy:
    1. Map input x from [-3, 3] to t in [-1, 1] via t = x / 3.
    2. Evaluate Chebyshev polynomials T_0(t) ... T_9(t).
    3. The output is the linear combination: y = sum_{k=0}^{9} c_k * T_k(t).
    
    This approach is structurally different from the current program because:
    - It uses a global orthogonal-like basis rather than local/non-linear components (Gaussians/Sines).
    - It avoids numerical instability associated with high-degree raw powers (x^9) by using the stable recurrence relation for Chebyshev polynomials.
    - It provides a well-conditioned linear system for BFGS to solve, as the parameters directly scale the basis functions without non-linear coupling between parameters.
    """
    # Ensure x is a numpy array and flatten for consistent processing
    x = np.asarray(x, dtype=float).ravel()
    
    # Map x from [-3, 3] to t in [-1, 1]
    # This scaling ensures the Chebyshev polynomials are evaluated within their natural domain
    t = x / 3.0
    
    # Initialize arrays for Chebyshev polynomials T_k(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    # We need 10 coefficients (degree 0 to 9), so we need T_0 to T_9
    # params[0] corresponds to T_0, ..., params[9] corresponds to T_9
    
    # Start with the first two polynomials
    T_k_minus_2 = np.ones_like(t)   # T_0
    T_k_minus_1 = t                  # T_1
    
    # Accumulate the result
    # y = c_0 * T_0 + c_1 * T_1 + ... + c_9 * T_9
    y = params[0] * T_k_minus_2 + params[1] * T_k_minus_1
    
    # Compute T_2 through T_9
    for k in range(2, 10):
        # Recurrence: T_k = 2*t*T_{k-1} - T_{k-2}
        T_k = 2.0 * t * T_k_minus_1 - T_k_minus_2
        
        y += params[k] * T_k
        
        # Update for next iteration
        T_k_minus_2 = T_k_minus_1
        T_k_minus_1 = T_k
    
    # Ensure output is finite
    # Chebyshev polynomials are bounded on [-1, 1], so this is mostly a safety net
    # for extreme parameter values that might cause overflow in the summation
    return np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
