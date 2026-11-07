"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers the target function using a 10-term Chebyshev polynomial basis.
    The basis consists of products of 1D Chebyshev polynomials T_k(x) and T_m(y)
    such that k + m <= 3. This provides a well-conditioned approximation space
    for smooth functions on [-1, 1] x [-1, 1].
    """
    # Simple deadline guard (though this function is very fast)
    start_time = time.time()
    
    try:
        # Compute Chebyshev polynomials of the first kind up to degree 4
        # T_0(t) = 1
        # T_1(t) = t
        # T_2(t) = 2t^2 - 1
        # T_3(t) = 4t^3 - 3t
        # T_4(t) = 8t^4 - 8t^2 + 1
        
        T0_x = np.ones_like(x)
        T1_x = x
        T2_x = 2.0 * x**2 - 1.0
        T3_x = 4.0 * x**3 - 3.0 * x
        T4_x = 8.0 * x**4 - 8.0 * x**2 + 1.0
        
        T0_y = np.ones_like(y)
        T1_y = y
        T2_y = 2.0 * y**2 - 1.0
        T3_y = 4.0 * y**3 - 3.0 * y
        T4_y = 8.0 * y**4 - 8.0 * y**2 + 1.0
        
        # Construct the 10 basis terms
        # Use a basis that allows higher individual degrees:
        # 0: T0(x)T0(y)
        # 1: T1(x)T0(y)
        # 2: T0(x)T1(y)
        # 3: T2(x)T0(y)
        # 4: T0(x)T2(y)
        # 5: T3(x)T0(y)
        # 6: T0(x)T3(y)
        # 7: T4(x)T0(y)
        # 8: T0(x)T4(y)
        # 9: T1(x)T1(y)
        
        term0 = T0_x * T0_y
        term1 = T1_x * T0_y
        term2 = T0_x * T1_y
        term3 = T2_x * T0_y
        term4 = T0_x * T2_y
        term5 = T3_x * T0_y
        term6 = T0_x * T3_y
        term7 = T4_x * T0_y
        term8 = T0_x * T4_y
        term9 = T1_x * T1_y
        
        prediction = (params[0] * term0 + 
                      params[1] * term1 + 
                      params[2] * term2 + 
                      params[3] * term3 + 
                      params[4] * term4 + 
                      params[5] * term5 + 
                      params[6] * term6 + 
                      params[7] * term7 + 
                      params[8] * term8 + 
                      params[9] * term9)
        
        # Check for non-finite values
        if not np.all(np.isfinite(prediction)):
             # Fallback to zero if something went wrong
             return np.zeros_like(x)
             
        return prediction
        
    except Exception:
        # In case of any error, return zeros to avoid crashing the evaluator
        return np.zeros_like(x)
# EVOLVE-BLOCK-END
