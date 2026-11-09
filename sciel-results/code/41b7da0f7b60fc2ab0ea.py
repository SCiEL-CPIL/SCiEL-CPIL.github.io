import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Chebyshev polynomials (global smoothness) + Gaussian RBFs (local features).
    
    The 10 parameters are split:
    - First 5: Weights for Chebyshev polynomials T_0 to T_4
    - Last 5: Weights for Gaussian RBFs centered at -1.5, -0.5, 0.5, 1.5, 0
    
    This structure allows the model to capture both global trends (via Chebyshev)
    and local variations or sharp features (via RBFs), which pure polynomial bases
    might miss if the target function has non-polynomial characteristics.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Ensure x is 1D
    if x.ndim != 1:
        x = x.flatten()
    
    n = x.shape[0]
    
    # Scale x to [-1, 1] for Chebyshev
    t = x / 2.0
    
    # Build basis matrix
    basis = np.empty((n, 10), dtype=np.float64)
    
    # Part 1: Chebyshev polynomials T_0 to T_4
    # T_0(t) = 1
    basis[:, 0] = 1.0
    
    # T_1(t) = t
    basis[:, 1] = t
    
    # Recurrence: T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    T_prev = np.ones_like(t)  # T_0
    T_curr = t.copy()         # T_1
    
    for k in range(2, 5):
        T_next = 2.0 * t * T_curr - T_prev
        basis[:, k] = T_next
        T_prev = T_curr
        T_curr = T_next
    
    # Part 2: Gaussian RBFs
    # Centers: -1.5, -0.5, 0.5, 1.5, 0
    centers = np.array([-1.5, -0.5, 0.5, 1.5, 0.0])
    sigma = 0.6
    
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    gaussians = np.exp(-0.5 * (diff / sigma) ** 2)
    
    basis[:, 5:10] = gaussians
    
    # Compute prediction
    result = basis @ params
    
    return result
