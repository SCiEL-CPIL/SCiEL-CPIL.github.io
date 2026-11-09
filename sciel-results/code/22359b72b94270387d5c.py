import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Chebyshev Polynomials of the first kind up to degree 3
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2 * x**2 - 1
    T3_x = 4 * x**3 - 3 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2 * y**2 - 1
    T3_y = 4 * y**3 - 3 * y
    
    # Construct basis functions: Complete tensor product up to degree 3
    # Degree 0:
    b1 = T0_x * T0_y  # 1
    
    # Degree 1:
    b2 = T1_x * T0_y  # x
    b3 = T0_x * T1_y  # y
    
    # Degree 2:
    b4 = T2_x * T0_y  # T2(x)
    b5 = T0_x * T2_y  # T2(y)
    b6 = T1_x * T1_y  # x*y
    
    # Degree 3:
    b7 = T3_x * T0_y  # T3(x)
    b8 = T0_x * T3_y  # T3(y)
    b9 = T2_x * T1_y  # T2(x)*y
    b10 = T1_x * T2_y  # x*T2(y)
    
    result = (params[0] * b1 +
              params[1] * b2 +
              params[2] * b3 +
              params[3] * b4 +
              params[4] * b5 +
              params[5] * b6 +
              params[6] * b7 +
              params[7] * b8 +
              params[8] * b9 +
              params[9] * b10)
    
    # Check deadline
    if time.time() > deadline:
        # Return best result found so far
        return result
    
    return result
