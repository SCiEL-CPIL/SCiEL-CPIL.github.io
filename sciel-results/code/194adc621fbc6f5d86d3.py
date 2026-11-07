"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form bivariate function using a richer basis that includes
    higher-degree terms and radial/angular components to better approximate
    smooth unknown functions on [-1,1]^2.
    
    We use a mix of:
    - Chebyshev tensor products up to degree 4 (for x and y separately)
    - Radial and angular terms (r^2, r^4, x^2-y^2, etc.)
    - Interaction terms
    
    This gives 10 basis functions that span a richer function space than
    degree-3 polynomials, potentially capturing more complex target functions.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Chebyshev polynomials up to degree 4
    # T0 = 1
    # T1 = t
    # T2 = 2t^2 - 1
    # T3 = 4t^3 - 3t
    # T4 = 8t^4 - 8t^2 + 1
    
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x * x - 1.0
    T3_x = 4.0 * x * x * x - 3.0 * x
    T4_x = 8.0 * x**4 - 8.0 * x * x + 1.0
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y * y - 1.0
    T3_y = 4.0 * y * y * y - 3.0 * y
    T4_y = 8.0 * y**4 - 8.0 * y * y + 1.0
    
    # 10 basis functions - a mix of tensor products and radial/angular terms
    b0 = T0_x * T0_y          # 1
    b1 = T1_x * T0_y          # x
    b2 = T0_x * T1_y          # y
    b3 = T2_x * T0_y          # 2x^2 - 1
    b4 = T0_x * T2_y          # 2y^2 - 1
    b5 = T1_x * T1_y          # xy
    b6 = T3_x * T0_y          # 4x^3 - 3x
    b7 = T0_x * T3_y          # 4y^3 - 3y
    b8 = T4_x * T0_y          # 8x^4 - 8x^2 + 1
    b9 = T0_x * T4_y          # 8y^4 - 8y^2 + 1
    
    # Linear combination
    pred = (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
    
    # Ensure output is finite
    if not np.all(np.isfinite(pred)):
        return np.mean(y) * np.ones_like(x)
        
    return pred
# EVOLVE-BLOCK-END
