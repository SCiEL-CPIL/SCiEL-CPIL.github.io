# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear model using Chebyshev polynomials of the first kind.
    The basis functions are T_0(x/3) to T_9(x/3).
    This formulation is linear in parameters, ensuring a convex
    optimization landscape for BFGS.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Scale x to [-1, 1] to utilize Chebyshev polynomials effectively
    # x is in [-3, 3], so x/3 is in [-1, 1]
    t = x / 3.0
    
    # Precompute Chebyshev polynomials T_0 through T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_n(t) = 2*t*T_{n-1}(t) - T_{n-2}(t)
    
    # We can compute the basis matrix directly
    # Shape: (10, N)
    N = len(x)
    basis = np.empty((10, N), dtype=np.float64)
    
    basis[0, :] = 1.0
    basis[1, :] = t
    
    for n in range(2, 10):
        basis[n, :] = 2.0 * t * basis[n-1, :] - basis[n-2, :]
        
    # Compute y = basis @ params
    y = basis @ params
    
    # Safety check for non-finite values
    y = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return y
# EVOLVE-BLOCK-END
