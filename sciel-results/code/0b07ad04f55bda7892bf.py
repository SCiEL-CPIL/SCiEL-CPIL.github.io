import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recover an unknown function using a mixed basis of polynomials, trigonometric,
    hyperbolic, and Gaussian terms. This provides greater expressive power than
    pure polynomials for diverse hidden functions.
    """
    start_time = time.time()
    x = np.asarray(x, dtype=np.float64)
    n_points = x.size
    
    # Define the 10 basis functions
    # 1. Constant
    # 2. Linear
    # 3. Quadratic
    # 4. Sine
    # 5. Cosine
    # 6. Tanh
    # 7. Gaussian at x=2
    # 8. Gaussian at x=-2
    # 9. Gaussian at x=0
    # 10. Sine(2x)
    
    T = np.empty((n_points, 10), dtype=np.float64)
    
    T[:, 0] = 1.0
    T[:, 1] = x
    T[:, 2] = x * x
    T[:, 3] = np.sin(x)
    T[:, 4] = np.cos(x)
    T[:, 5] = np.tanh(x)
    T[:, 6] = np.exp(-(x - 2.0) ** 2)
    T[:, 7] = np.exp(-(x + 2.0) ** 2)
    T[:, 8] = np.exp(-x * x)
    T[:, 9] = np.sin(2.0 * x)
    
    # Check for non-finite values just in case, though inputs are from [-3,3]
    if not np.all(np.isfinite(T)):
         # Fallback to simple polynomial if numerical issues arise (unlikely)
         T[:, 0] = 1.0
         T[:, 1] = x
         T[:, 2] = x**2
         T[:, 3] = x**3
         T[:, 4] = x**4
         T[:, 5] = x**5
         T[:, 6] = x**6
         T[:, 7] = x**7
         T[:, 8] = x**8
         T[:, 9] = x**9
         
    # Linear combination
    return T @ params
