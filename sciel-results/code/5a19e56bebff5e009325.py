# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a sum of 10 basis functions.
    The 10 parameters are the weights for each basis function.
    
    Basis functions are chosen to be smooth and well-conditioned on [-2, 2]:
    - 5 Chebyshev polynomials T_0, T_1, T_2, T_3, T_4 (mapped to [-1,1])
    - 5 Gaussian bumps centered at -1.5, -0.75, 0, 0.75, 1.5
    
    This combination allows modeling both global polynomial trends and local features.
    """
    n = len(x)
    if n == 0:
        return np.array([])
    
    # Map x from [-2, 2] to [-1, 1] for Chebyshev polynomials
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 through T_4
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    basis = np.zeros((10, n), dtype=np.float64)
    
    # Chebyshev basis (first 5)
    basis[0, :] = 1.0
    if n > 0:
        basis[1, :] = t
        for k in range(2, 5):
            basis[k, :] = 2.0 * t * basis[k-1, :] - basis[k-2, :]
    
    # Gaussian bumps (last 5)
    # Centers at -1.5, -0.75, 0, 0.75, 1.5 with width 1.0
    centers = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])
    width = 1.0
    
    for i, c in enumerate(centers):
        basis[5 + i, :] = np.exp(-((x - c) ** 2) / (2.0 * width ** 2))
    
    # Compute linear combination: params @ basis.T
    # params shape: (10,), basis shape: (10, n)
    result = basis.T @ params
    
    # Ensure output is finite
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    return result
# EVOLVE-BLOCK-END
