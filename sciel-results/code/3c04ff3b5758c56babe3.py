import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic Regression using Chebyshev Polynomial Basis.
    
    Strategy:
    - Use Chebyshev polynomials of the first kind (T_0 to T_9) as the basis.
    - Chebyshev polynomials are orthogonal on [-1, 1] and well-conditioned on bounded intervals.
    - This makes the design matrix well-conditioned, allowing BFGS to converge 
      reliably to the global minimum for the linear-in-parameters model.
    - The domain [0.1, 6] is mapped to [-1, 1] to ensure optimal conditioning.
    
    Parameters:
    - p[0] to p[9]: Coefficients for T_0(x) to T_9(x).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    p = np.asarray(params, dtype=np.float64)
    if p.ndim != 1 or p.shape[0] != 10:
        return np.zeros_like(x)
    
    # Map x from [0.1, 6] to t in [-1, 1]
    # t = 2 * (x - min) / (max - min) - 1
    # min = 0.1, max = 6
    # t = 2 * (x - 0.1) / 5.9 - 1 = (2x - 0.2 - 5.9) / 5.9 = (2x - 6.1) / 5.9
    # To be safe and general, we can use the exact bounds or a slightly wider range.
    # Using exact bounds is fine for interpolation/fitting within the range.
    x_min = 0.1
    x_max = 6.0
    t = (2.0 * x - (x_max + x_min)) / (x_max - x_min)
    
    # Compute Chebyshev polynomials T_0(t) to T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2*t*T_n(t) - T_{n-1}(t)
    
    T = np.zeros((10, len(x)))
    T[0, :] = 1.0
    if len(x) > 0:
        T[1, :] = t
    for n in range(2, 10):
        T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
    
    # Linear combination
    # result = sum(p[i] * T[i, :] for i in range(10))
    # This can be done via matrix multiplication: T @ p
    # But T is (10, N) and p is (10,), so we need T.T @ p
    result = T.T @ p
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result
