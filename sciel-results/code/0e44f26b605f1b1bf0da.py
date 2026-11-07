"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def chebyshev_basis_2d(x, y):
    """
    Returns an array of shape (N, 10) with Chebyshev basis functions up to degree 3.
    Terms:
    1. 1
    2. x
    3. y
    4. 2x^2 - 1
    5. xy
    6. 2y^2 - 1
    7. 4x^3 - 3x
    8. (2x^2 - 1)y
    9. x(2y^2 - 1)
    10. 4y^3 - 3y
    """
    N = len(x)
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Precompute powers
    x2 = x * x
    x3 = x2 * x
    y2 = y * y
    y3 = y2 * y
    
    # Chebyshev polynomials
    T0 = np.ones(N)
    T1_x = x
    T2_x = 2 * x2 - 1
    T3_x = 4 * x3 - 3 * x
    
    T1_y = y
    T2_y = 2 * y2 - 1
    T3_y = 4 * y3 - 3 * y
    
    basis = np.column_stack([
        T0,                  # 1
        T1_x,                # x
        T1_y,                # y
        T2_x,                # 2x^2 - 1
        T1_x * T1_y,         # xy
        T2_y,                # 2y^2 - 1
        T3_x,                # 4x^3 - 3x
        T2_x * T1_y,         # (2x^2 - 1)y
        T1_x * T2_y,         # x(2y^2 - 1)
        T3_y                 # 4y^3 - 3y
    ])
    
    return basis

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
