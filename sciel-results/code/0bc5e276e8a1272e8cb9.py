import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a Chebyshev polynomial basis.
    Chebyshev polynomials are orthogonal on [-1, 1] and provide a stable
    representation for smooth functions, including polynomials up to degree 3.
    
    Basis:
    T0(x) = 1
    T1(x) = x
    T2(x) = 2x^2 - 1
    T3(x) = 4x^3 - 3x
    
    The model is a linear combination of products of these bases:
    f(x, y) = p0*T0(x)T0(y) + p1*T1(x)T0(y) + p2*T0(x)T1(y) 
             + p3*T2(x)T0(y) + p4*T0(x)T2(y) + p5*T1(x)T1(y)
             + p6*T3(x)T0(y) + p7*T0(x)T3(y) + p8*T2(x)T1(y) + p9*T1(x)T2(y)
    """
    # Compute Chebyshev polynomials for x and y
    # T0 = 1
    # T1 = x
    # T2 = 2x^2 - 1
    # T3 = 4x^3 - 3x
    
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2 * x * x - 1
    T3_x = 4 * x * x * x - 3 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2 * y * y - 1
    T3_y = 4 * y * y * y - 3 * y
    
    # Construct prediction
    # Order: 
    # p0: T0_x * T0_y
    # p1: T1_x * T0_y
    # p2: T0_x * T1_y
    # p3: T2_x * T0_y
    # p4: T0_x * T2_y
    # p5: T1_x * T1_y
    # p6: T3_x * T0_y
    # p7: T0_x * T3_y
    # p8: T2_x * T1_y
    # p9: T1_x * T2_y
    
    pred = (params[0] * T0_x * T0_y 
            + params[1] * T1_x * T0_y 
            + params[2] * T0_x * T1_y 
            + params[3] * T2_x * T0_y 
            + params[4] * T0_x * T2_y 
            + params[5] * T1_x * T1_y 
            + params[6] * T3_x * T0_y 
            + params[7] * T0_x * T3_y 
            + params[8] * T2_x * T1_y 
            + params[9] * T1_x * T2_y)
            
    return pred
