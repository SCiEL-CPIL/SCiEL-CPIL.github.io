import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the target function using a rational function with Chebyshev basis.
    f(x) = N(x) / D(x)
    N(x) = sum_{i=0}^4 a_i T_i(x/2)
    D(x) = sum_{j=0}^4 b_j T_j(x/2)
    
    params = [a0, a1, a2, a3, a4, b0, b1, b2, b3, b4]
    """
    x = np.asarray(x, dtype=float)
    
    # Scale x from [-2, 2] to [-1, 1]
    t = 0.5 * x
    
    # Compute Chebyshev polynomials T_0 to T_4
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2 * t * t - 1
    T3 = 4 * t**3 - 3 * t
    T4 = 8 * t**4 - 8 * t**2 + 1
    
    # Split params into numerator and denominator coefficients
    a = params[0:5]  # a0, a1, a2, a3, a4
    b = params[5:10]  # b0, b1, b2, b3, b4
    
    # Compute numerator and denominator
    N = a[0] * T0 + a[1] * T1 + a[2] * T2 + a[3] * T3 + a[4] * T4
    D = b[0] * T0 + b[1] * T1 + b[2] * T2 + b[3] * T3 + b[4] * T4
    
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-10
    D_safe = np.where(np.abs(D) < epsilon, epsilon, D)
    
    # Return the rational function
    result = N / D_safe
    
    return result
