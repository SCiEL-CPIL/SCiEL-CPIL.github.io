import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Fixed, well-conditioned linear basis using Chebyshev polynomials.
    
    This is a structural departure from non-linear parametric models or adaptive
    basis selection. By using a fixed linear model y = sum p_i * T_i(x), the
    optimization problem becomes convex (linear least squares). BFGS will converge
    rapidly and robustly to the global optimum for the given basis, avoiding
    local minima and conditioning issues associated with non-linear parameter
    searches.
    
    Chebyshev polynomials are used because they are orthogonal on [-1, 1] and
    well-conditioned, providing a stable basis for approximating smooth functions
    on [-2, 2] (mapped to [-1, 1]).
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(params, dtype=float)
    
    # Safety check
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(p)):
        return np.zeros_like(x)
        
    # Map x from [-2, 2] to [-1, 1] for Chebyshev polynomials
    x_norm = 0.5 * x
    
    # Compute Chebyshev polynomials T_0 to T_9
    n = 10
    T = np.zeros((len(x), n))
    T[:, 0] = 1.0
    if len(x) > 0:
        T[:, 1] = x_norm
    for k in range(2, n):
        T[:, k] = 2.0 * x_norm * T[:, k-1] - T[:, k-2]
        
    # Linear combination
    out = T @ p
    
    # Ensure finite
    if not np.all(np.isfinite(out)):
        out = np.where(np.isfinite(out), out, 0.0)
        
    return out
