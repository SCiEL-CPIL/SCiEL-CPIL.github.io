"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 Chebyshev polynomials of the first kind.
    
    The 10 parameters represent the coefficients of Chebyshev polynomials T_0 to T_9.
    The input x is mapped from [-2, 2] to [-1, 1] to match the standard domain of Chebyshev polynomials.
    
    This basis is orthogonal and provides a stable, global approximation for smooth functions,
    avoiding the conditioning issues of monomials and the locality constraints of RBFs.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Map x from [-2, 2] to [-1, 1]
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0(t) ... T_9(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_k(t) = 2*t*T_{k-1}(t) - T_{k-2}(t)
    
    N = len(x)
    basis = np.empty((N, 10), dtype=np.float64)
    
    basis[:, 0] = 1.0
    if N > 0:
        basis[:, 1] = t
        for k in range(2, 10):
            basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
            
    # Linear combination: result = basis @ params
    result = basis @ params
    
    # Safety: replace non-finite values
    result = np.where(np.isfinite(result), result, 1e6)
    
    return result
# EVOLVE-BLOCK-END
