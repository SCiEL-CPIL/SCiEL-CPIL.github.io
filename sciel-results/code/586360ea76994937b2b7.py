"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute Chebyshev polynomials of the first kind
# T_0(t) = 1
# T_1(t) = t
# T_2(t) = 2t^2 - 1
# T_3(t) = 4t^3 - 3t

def chebyshev_basis_2d(x, y):
    """
    Returns an array of shape (N, 10) with the 10 Chebyshev basis functions
    for total degree <= 3 in 2D.
    """
    N = len(x)
    # Compute 1D Chebyshev values
    T0_x = np.ones(N)
    T1_x = x
    T2_x = 2 * x**2 - 1
    T3_x = 4 * x**3 - 3 * x
    
    T0_y = np.ones(N)
    T1_y = y
    T2_y = 2 * y**2 - 1
    T3_y = 4 * y**3 - 3 * y
    
    # Construct the 10 terms for total degree <= 3
    # (0,0), (1,0), (2,0), (3,0)
    # (0,1), (1,1), (2,1)
    # (0,2), (1,2)
    # (0,3)
    
    terms = np.zeros((N, 10))
    terms[:, 0] = T0_x * T0_y
    terms[:, 1] = T1_x * T0_y
    terms[:, 2] = T2_x * T0_y
    terms[:, 3] = T3_x * T0_y
    
    terms[:, 4] = T0_x * T1_y
    terms[:, 5] = T1_x * T1_y
    terms[:, 6] = T2_x * T1_y
    
    terms[:, 7] = T0_x * T2_y
    terms[:, 8] = T1_x * T2_y
    
    terms[:, 9] = T0_x * T3_y
    
    return terms


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Check if params are valid
    if not np.all(np.isfinite(params)):
        return np.zeros(len(x))
        
    # Compute basis
    basis = chebyshev_basis_2d(x, y)
    
    # Predict
    pred = basis @ params
    
    # Ensure finite
    if not np.all(np.isfinite(pred)):
        return np.zeros(len(x))
        
    return pred
# EVOLVE-BLOCK-END
