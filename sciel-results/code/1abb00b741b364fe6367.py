import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fits a Legendre polynomial series of degree 9 to the data.
    Input x is in [-2, 2]. We scale to t in [-1, 1] for Legendre stability.
    params contains 10 coefficients for P_0(t) through P_9(t).
    Legendre polynomials are orthogonal with respect to the uniform weight,
    which matches the uniform distribution of the input data.
    """
    # Scale x from [-2, 2] to [-1, 1]
    t = x / 2.0
    
    # Compute Legendre polynomials P_0(t) to P_9(t)
    # P_0(t) = 1
    # P_1(t) = t
    # P_{n+1}(t) = ((2n+1)*t*P_n(t) - n*P_{n-1}(t)) / (n+1)
    
    P_0 = np.ones_like(t)
    P_1 = t.copy()
    
    result = params[0] * P_0 + params[1] * P_1
    
    if len(params) > 2:
        P_prev = P_0
        P_curr = P_1
        for k in range(2, len(params)):
            # n = k-1
            # P_k = ((2(k-1)+1)*t*P_{k-1} - (k-1)*P_{k-2}) / k
            n = k - 1
            P_next = ((2 * n + 1) * t * P_curr - n * P_prev) / k
            result += params[k] * P_next
            P_prev = P_curr
            P_curr = P_next
            
    return result
