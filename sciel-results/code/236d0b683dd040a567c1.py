"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x from [0.1, 6] to [-1, 1] for better numerical stability
    # z = (2 * (x - min) / (max - min)) - 1
    # min = 0.1, max = 6.0
    # z = (2 * (x - 0.1) / 5.9) - 1 = (2x - 0.2)/5.9 - 1 = (2x - 5.9)/5.9
    
    # Calculate scaled variable z
    z = (2.0 * x - 5.9) / 5.9
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(z) = 1
    # T_1(z) = z
    # T_n(z) = 2z T_{n-1}(z) - T_{n-2}(z)
    
    T = np.zeros((10, len(z)))
    T[0, :] = 1.0
    T[1, :] = z
    
    for n in range(2, 10):
        T[n, :] = 2.0 * z * T[n-1, :] - T[n-2, :]
        
    # Combine with parameters
    # params is shape (10,)
    # T is shape (10, N)
    # result = T.T @ params
    return T.T @ params
# EVOLVE-BLOCK-END
