import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Chebyshev polynomial basis of the first kind
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2t^2 - 1
    # T3(t) = 4t^3 - 3t
    # T4(t) = 8t^4 - 8t^2 + 1
    
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2 * x**2 - 1
    T3_x = 4 * x**3 - 3 * x
    T4_x = 8 * x**4 - 8 * x**2 + 1
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2 * y**2 - 1
    T3_y = 4 * y**3 - 3 * y
    T4_y = 8 * y**4 - 8 * y**2 + 1
    
    # Basis vector with 10 terms:
    # 0: T0(x)*T0(y) = 1
    # 1: T1(x)*T0(y) = x
    # 2: T0(x)*T1(y) = y
    # 3: T2(x)*T0(y) = 2x^2 - 1
    # 4: T0(x)*T2(y) = 2y^2 - 1
    # 5: T1(x)*T1(y) = x*y
    # 6: T2(x)*T1(y) = (2x^2-1)*y
    # 7: T1(x)*T2(y) = x*(2y^2-1)
    # 8: T3(x)*T0(y) = 4x^3 - 3x
    # 9: T0(x)*T3(y) = 4y^3 - 3y
    
    f = (params[0] * T0_x * T0_y +
         params[1] * T1_x * T0_y +
         params[2] * T0_x * T1_y +
         params[3] * T2_x * T0_y +
         params[4] * T0_x * T2_y +
         params[5] * T1_x * T1_y +
         params[6] * T2_x * T1_y +
         params[7] * T1_x * T2_y +
         params[8] * T3_x * T0_y +
         params[9] * T0_x * T3_y)
         
    return f
