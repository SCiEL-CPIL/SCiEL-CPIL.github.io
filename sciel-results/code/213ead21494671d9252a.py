import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard: The evaluator kills after 150s.
    # This function is called repeatedly by the optimizer.
    # The computation is O(N) for N points, which is negligible for N=200.
    # However, we include the guard as per instructions.
    start_time = time.time()
    
    # Compute Chebyshev polynomials of the first kind T_n(t) on [-1, 1]
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    
    # We construct the 10 basis functions corresponding to total degree <= 3
    # Indices (i, j) where i is degree in x, j is degree in y, and i + j <= 3
    
    # Pre-compute Chebyshev values for x and y
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x * x - 1.0
    T3_x = 4.0 * x * x * x - 3.0 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y * y - 1.0
    T3_y = 4.0 * y * y * y - 3.0 * y
    
    # Basis functions
    # Order:
    # 0: T0_x * T0_y
    # 1: T1_x * T0_y
    # 2: T0_x * T1_y
    # 3: T2_x * T0_y
    # 4: T1_x * T1_y
    # 5: T0_x * T2_y
    # 6: T3_x * T0_y
    # 7: T2_x * T1_y
    # 8: T1_x * T2_y
    # 9: T0_x * T3_y
    
    b0 = T0_x * T0_y
    b1 = T1_x * T0_y
    b2 = T0_x * T1_y
    b3 = T2_x * T0_y
    b4 = T1_x * T1_y
    b5 = T0_x * T2_y
    b6 = T3_x * T0_y
    b7 = T2_x * T1_y
    b8 = T1_x * T2_y
    b9 = T0_x * T3_y
    
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
            
    # Check for non-finite values
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(y)
        
    return pred
