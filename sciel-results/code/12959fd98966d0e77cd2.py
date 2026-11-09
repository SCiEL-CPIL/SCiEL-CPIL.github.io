"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard: ensure we return within 150s wall-clock
    t_start = time.time()
    deadline = 145.0  # 5s safety margin
    
    # Construct 10 basis functions.
    # We use a mix of low-degree monomials and higher-degree pure monomials.
    # This allows capturing curvature (degree 2), skewness (degree 3), 
    # and 4th-order effects (degree 4) in the pure variables, which are 
    # common in smooth functions like sin, cos, exp on [-1, 1].
    # We drop the degree-3 cross terms (x^2*y, x*y^2) to make room for x^4, y^4.
    
    b0 = np.ones_like(x)       # 1
    b1 = x                     # x
    b2 = y                     # y
    b3 = x**2                  # x^2
    b4 = y**2                  # y^2
    b5 = x * y                 # x*y
    b6 = x**3                  # x^3
    b7 = y**3                  # y^3
    b8 = x**4                  # x^4
    b9 = y**4                  # y^4
    
    pred = (params[0] * b0 + params[1] * b1 + params[2] * b2 + 
            params[3] * b3 + params[4] * b4 + params[5] * b5 + 
            params[6] * b6 + params[7] * b7 + params[8] * b8 + 
            params[9] * b9)
    
    # Check deadline before returning
    if time.time() - t_start > deadline:
        return pred
    
    return pred
# EVOLVE-BLOCK-END
