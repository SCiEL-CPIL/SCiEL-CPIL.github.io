import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Chebyshev Polynomials of the first kind up to degree 4
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    # T_4(t) = 8t^4 - 8t^2 + 1
    
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
    
    # Construct basis functions
    # 1. Constant
    b1 = T0_x * T0_y
    
    # 2. Linear x
    b2 = T1_x * T0_y
    
    # 3. Linear y
    b3 = T0_x * T1_y
    
    # 4. Quadratic x (Chebyshev)
    b4 = T2_x * T0_y
    
    # 5. Quadratic y (Chebyshev)
    b5 = T0_x * T2_y
    
    # 6. Interaction xy
    b6 = T1_x * T1_y
    
    # 7. Cubic x (Chebyshev)
    b7 = T3_x * T0_y
    
    # 8. Cubic y (Chebyshev)
    b8 = T0_x * T3_y
    
    # 9. Quartic x (Chebyshev)
    b9 = T4_x * T0_y
    
    # 10. Quartic y (Chebyshev)
    b10 = T0_x * T4_y
    
    return (params[0] * b1 +
            params[1] * b2 +
            params[2] * b3 +
            params[3] * b4 +
            params[4] * b5 +
            params[5] * b6 +
            params[6] * b7 +
            params[7] * b8 +
            params[8] * b9 +
            params[9] * b10)
