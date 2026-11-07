import numpy as np

def equation(x, y, params):
    # Chebyshev basis up to degree 4 for better approximation
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

    # Basis functions: include higher order terms
    # Using 10 params to cover more of the function space
    b0 = T0_x * T0_y
    b1 = T1_x * T0_y
    b2 = T0_x * T1_y
    b3 = T2_x * T0_y
    b4 = T0_x * T2_y
    b5 = T1_x * T1_y
    b6 = T3_x * T0_y
    b7 = T0_x * T3_y
    b8 = T2_x * T2_y
    b9 = T1_x * T2_y

    return params[0]*b0 + params[1]*b1 + params[2]*b2 + params[3]*b3 + \
           params[4]*b4 + params[5]*b5 + params[6]*b6 + params[7]*b7 + \
           params[8]*b8 + params[9]*b9
