"""Symbolic regression with a hybrid Chebyshev + Gaussian RBF basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: The evaluator kills after 150s wall-clock.
_START_TIME = time.time()
_MAX_TIME = 145.0


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis function combining Chebyshev polynomials and Gaussian RBFs.
    
    The input x is in [-3, 3]. We normalize x to [-1, 1] for Chebyshev stability.
    We use 5 Chebyshev polynomials (T_0 to T_4) to capture global smooth trends,
    and 5 Gaussian RBFs centered at {-3, -1.5, 0, 1.5, 3} to capture local features.
    
    Basis:
    0: T_0(z) = 1
    1: T_1(z) = z
    2: T_2(z) = 2z^2 - 1
    3: T_3(z) = 4z^3 - 3z
    4: T_4(z) = 8z^4 - 8z^2 + 1
    5: exp(-((x - (-3))^2) / (2 * sigma^2))
    6: exp(-((x - (-1.5))^2) / (2 * sigma^2))
    7: exp(-((x - 0)^2) / (2 * sigma^2))
    8: exp(-((x - 1.5)^2) / (2 * sigma^2))
    9: exp(-((x - 3)^2) / (2 * sigma^2))
    
    The RBF width sigma is chosen to provide overlap between centers,
    allowing smooth interpolation while maintaining locality.
    """
    # Check time limit
    if time.time() - _START_TIME > _MAX_TIME:
        # Return a safe default if we are about to time out
        return np.zeros_like(x, dtype=np.float64)

    x = np.asarray(x, dtype=np.float64)
    
    # Normalize x for Chebyshev basis
    z = x / 3.0
    
    # Compute Chebyshev polynomials T_0 to T_4 using recurrence
    # T_0(z) = 1
    # T_1(z) = z
    # T_n(z) = 2*z*T_{n-1}(z) - T_{n-2}(z)
    
    T0 = np.ones_like(x)
    T1 = z
    T2 = 2.0 * z * T1 - T0
    T3 = 2.0 * z * T2 - T1
    T4 = 2.0 * z * T3 - T2
    
    # Gaussian RBFs with centers at {-3, -1.5, 0, 1.5, 3}
    # Width sigma = 1.5 provides good overlap for smooth functions
    sigma = 1.5
    sigma_sq_2 = 2.0 * sigma * sigma
    
    centers = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    # Compute all 5 RBFs efficiently
    # diff is (5, N)
    diff = (x[np.newaxis, :] - centers[:, np.newaxis])
    rbf = np.exp(-diff * diff / sigma_sq_2)
    
    # Linear combination
    # params is length 10
    pred = (params[0] * T0 + 
            params[1] * T1 + 
            params[2] * T2 + 
            params[3] * T3 + 
            params[4] * T4 + 
            params[5] * rbf[0] +
            params[6] * rbf[1] +
            params[7] * rbf[2] +
            params[8] * rbf[3] +
            params[9] * rbf[4])
            
    return pred
# EVOLVE-BLOCK-END
