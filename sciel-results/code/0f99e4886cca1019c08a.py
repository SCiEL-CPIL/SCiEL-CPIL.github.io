import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = x.flatten()
    y = y.flatten()
    
    # Chebyshev product basis T_i(x)T_j(y) with i+j <= 3 (10 terms)
    # This provides an orthogonal, well-conditioned design matrix
    # compared to monomials, enabling BFGS to converge more accurately.
    
    # Compute Chebyshev polynomials up to degree 3 for x and y
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
    
    # Build 10 basis functions: T_i(x) * T_j(y) where i + j <= 3
    # Order:
    # 0: T0(x)*T0(y) = 1
    # 1: T1(x)*T0(y) = x
    # 2: T0(x)*T1(y) = y
    # 3: T2(x)*T0(y) = 2x^2 - 1
    # 4: T1(x)*T1(y) = x*y
    # 5: T0(x)*T2(y) = 2y^2 - 1
    # 6: T3(x)*T0(y) = 4x^3 - 3x
    # 7: T2(x)*T1(y) = (2x^2-1)*y
    # 8: T1(x)*T2(y) = x*(2y^2-1)
    # 9: T0(x)*T3(y) = 4y^3 - 3y
    
    features = np.column_stack([
        T0_x * T0_y,
        T1_x * T0_y,
        T0_x * T1_y,
        T2_x * T0_y,
        T1_x * T1_y,
        T0_x * T2_y,
        T3_x * T0_y,
        T2_x * T1_y,
        T1_x * T2_y,
        T0_x * T3_y
    ])
    
    return features @ params
