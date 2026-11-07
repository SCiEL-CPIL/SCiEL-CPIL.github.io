import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression with a 2D Chebyshev basis of total degree 3.
    The 10 basis functions are well-conditioned on [-1,1] and span all
    bivariate polynomials up to total degree 3, making the linear model
    easy for BFGS to fit from the all-ones start.
    """
    # Chebyshev polynomials of the first kind, evaluated at x and y
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x**2 - 1.0
    T3_x = 4.0 * x**3 - 3.0 * x

    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y**2 - 1.0
    T3_y = 4.0 * y**3 - 3.0 * y

    # Tensor-product basis, total degree <= 3 (10 terms)
    # Index: (deg_x, deg_y)
    # 0: (0,0)
    # 1: (1,0)
    # 2: (0,1)
    # 3: (2,0)
    # 4: (0,2)
    # 5: (1,1)
    # 6: (3,0)
    # 7: (0,3)
    # 8: (2,1)
    # 9: (1,2)

    f = (params[0] * T0_x * T0_y +
         params[1] * T1_x * T0_y +
         params[2] * T0_x * T1_y +
         params[3] * T2_x * T0_y +
         params[4] * T0_x * T2_y +
         params[5] * T1_x * T1_y +
         params[6] * T3_x * T0_y +
         params[7] * T0_x * T3_y +
         params[8] * T2_x * T1_y +
         params[9] * T1_x * T2_y)

    return f
