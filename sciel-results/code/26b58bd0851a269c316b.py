"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard for the entire evaluation process is handled by the evaluator,
# but we ensure this function is fast to minimize overhead in the BFGS loop.
# The BFGS optimizer calls this function many times.

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis using 5 Chebyshev polynomials (global trend) 
    and 5 Gaussian RBFs (local features).
    
    This is a linear model in the 10 parameters, ensuring BFGS 
    converges to the global minimum efficiently.
    
    - Chebyshev T_0 to T_4 on x in [-3, 3] mapped to [-1, 1].
    - Gaussian RBFs at centers -3, -1.5, 0, 1.5, 3 with sigma=1.0.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Normalize x to [-1, 1] for Chebyshev stability
    t = x / 3.0
    
    # Chebyshev Polynomials T_0 to T_4
    b0 = np.ones_like(x)
    b1 = t
    b2 = 2.0 * t * t - 1.0
    b3 = 4.0 * b1 * b1 * b1 - 3.0 * b1
    b4 = 8.0 * b1 * b1 * b1 * b1 - 8.0 * b1 * b1 + 1.0
    
    # Gaussian RBFs
    # Centers distributed across the domain
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    sigma = 1.0
    
    # Compute RBFs vectorized: shape (N, 5)
    # diff[i, j] = x[i] - centers[j]
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Linear combination
    # params[0:5] for Chebyshev, params[5:10] for RBFs
    y = (params[0] * b0 + 
         params[1] * b1 + 
         params[2] * b2 + 
         params[3] * b3 + 
         params[4] * b4)
         
    # Add RBF contributions
    y += np.sum(rbf * params[5:10], axis=1)
         
    return y
# EVOLVE-BLOCK-END
