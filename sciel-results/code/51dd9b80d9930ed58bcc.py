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
    # T_0 = 1
    # T_1 = t
    # T_2 = 2*t^2 - 1
    # T_3 = 4*t^3 - 3*t
    # T_4 = 8*t^4 - 8*t^2 + 1
    
    t2 = t * t
    t3 = t2 * t
    t4 = t2 * t2
    
    T = np.stack([
        np.ones_like(t),
        t,
        2.0 * t2 - 1.0,
        4.0 * t3 - 3.0 * t,
        8.0 * t4 - 8.0 * t2 + 1.0
    ], axis=-1)
    
    # T is shape (N, 5)
    
    # Split params
    a = params[:5]
    b = params[5:10]
    
    # Compute Numerator and Denominator
    N = T @ a
    D = T @ b
    
    # Handle potential division by zero
    # If D is very close to 0, the prediction will be huge.
    # To avoid NaN/Inf, we clip the denominator.
    # Using a small epsilon to prevent exact zero.
    eps = 1e-8
    # Preserve sign
    D_safe = np.where(np.abs(D) < eps, np.sign(D) * eps, D)
    # If D is 0, sign is 0, so sign(D)*eps is 0. We need to handle 0 explicitly.
    D_safe = np.where(D == 0, eps, D_safe)
    
    result = N / D_safe
    
    return result
