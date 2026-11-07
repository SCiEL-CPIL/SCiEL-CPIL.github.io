import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using an orthogonal basis of Legendre polynomials
    up to degree 3 in each variable.
    
    The basis spans the space of polynomials of total degree <= 3.
    There are exactly 10 such monomials:
    1, x, y, x^2, y^2, xy, x^3, y^3, x^2y, xy^2.
    
    Using Legendre polynomials ensures the basis functions are orthogonal
    over the domain [-1, 1]^2, which leads to a well-conditioned least-squares
    problem. This allows scipy.optimize.minimize (BFGS) to converge reliably
    and quickly to the optimal parameters, even if the target is a polynomial
    of degree 3.
    
    Basis terms:
    0: P0(x)P0(y) = 1
    1: P1(x)P0(y) = x
    2: P0(x)P1(y) = y
    3: P2(x)P0(y) = 0.5 * (3x^2 - 1)
    4: P0(x)P2(y) = 0.5 * (3y^2 - 1)
    5: P1(x)P1(y) = x * y
    6: P3(x)P0(y) = 0.5 * (5x^3 - 3x)
    7: P0(x)P3(y) = 0.5 * (5y^3 - 3y)
    8: P2(x)P1(y) = 0.5 * (3x^2 - 1) * y
    9: P1(x)P2(y) = x * 0.5 * (3y^2 - 1)
    """
    # Precompute Legendre polynomials for x and y
    # P0(t) = 1
    # P1(t) = t
    # P2(t) = 0.5 * (3t^2 - 1)
    # P3(t) = 0.5 * (5t^3 - 3t)
    
    P0_x = np.ones_like(x)
    P1_x = x
    P2_x = 0.5 * (3 * x**2 - 1)
    P3_x = 0.5 * (5 * x**3 - 3 * x)
    
    P0_y = np.ones_like(y)
    P1_y = y
    P2_y = 0.5 * (3 * y**2 - 1)
    P3_y = 0.5 * (5 * y**3 - 3 * y)
    
    # Construct the 10 orthogonal basis functions
    b0 = P0_x * P0_y
    b1 = P1_x * P0_y
    b2 = P0_x * P1_y
    b3 = P2_x * P0_y
    b4 = P0_x * P2_y
    b5 = P1_x * P1_y
    b6 = P3_x * P0_y
    b7 = P0_x * P3_y
    b8 = P2_x * P1_y
    b9 = P1_x * P2_y
    
    # Linear combination
    f = (params[0] * b0 +
         params[1] * b1 +
         params[2] * b2 +
         params[3] * b3 +
         params[4] * b4 +
         params[5] * b5 +
         params[6] * b6 +
         params[7] * b7 +
         params[8] * b8 +
         params[9] * b9)
    
    return f
