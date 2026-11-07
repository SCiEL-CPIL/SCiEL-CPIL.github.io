# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear combination of 10 fixed basis functions.
    Basis: 1, x, y, z, x^2, y^2, z^2, x*y, x*z, y*z
    (degree-0, degree-1, degree-2 polynomial basis in 3 variables)
    params[0..9] are the coefficients.
    """
    # 10 basis features (all degree <= 2, symmetric)
    # This gives a quadratic polynomial in 3 variables:
    # f = p0 + p1*x + p2*y + p3*z + p4*x^2 + p5*y^2 + p6*z^2 + p7*x*y + p8*x*z + p9*y*z
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = z
    b4 = x * x
    b5 = y * y
    b6 = z * z
    b7 = x * y
    b8 = x * z
    b9 = y * z
    
    return (params[0] * b0 + params[1] * b1 + params[2] * b2 + params[3] * b3 +
            params[4] * b4 + params[5] * b5 + params[6] * b6 +
            params[7] * b7 + params[8] * b8 + params[9] * b9)
# EVOLVE-BLOCK-END
