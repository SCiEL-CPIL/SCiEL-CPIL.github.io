import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Chebyshev polynomial expansion of degree 9.
    
    The model uses the first 10 Chebyshev polynomials of the first kind, T_0(x) through T_9(x),
    mapped to the domain [-1, 1]. The input x is uniformly distributed in [-2, 2], so we map
    x in [-2, 2] to t in [-1, 1] via t = x / 2.
    
    y(x) = sum_{k=0}^{9} params[k] * T_k(x/2)
    
    This provides a stable, global basis that efficiently captures high-frequency oscillations
    and complex smooth shapes without the numerical instability of high-degree monomials or the
    local minima issues of localized non-linear units like tanh or RBFs.
    """
    # Map x from [-2, 2] to t in [-1, 1]
    t = x / 2.0
    
    # Ensure t is within [-1, 1] to avoid numerical issues, though x is guaranteed in [-2, 2]
    # t = np.clip(t, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_0(t) to T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2*t*T_n(t) - T_{n-1}(t)
    
    T0 = np.ones_like(t)
    T1 = t
    
    # We need T_0 to T_9. We have T_0 and T_1.
    # We will store them in a list or compute directly.
    # Since params length is 10, we need 10 basis functions.
    
    # Let's build the basis matrix B where B[i, j] = T_j(x_i)
    # x is 1D array. We can compute column by column.
    
    # Initialize result
    y = params[0] * T0 + params[1] * T1
    
    # Compute T_2 to T_9
    T_prev = T0
    T_curr = T1
    
    for k in range(2, 10):
        T_next = 2.0 * t * T_curr - T_prev
        y += params[k] * T_next
        T_prev = T_curr
        T_curr = T_next
        
    return y
