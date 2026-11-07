import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluates a mixed Chebyshev and linear basis with scaled input.
    
    The model uses 10 parameters:
        f(x) = params[0] * 1
              + params[1] * T_1(z)
              + params[2] * T_2(z)
              + params[3] * T_3(z)
              + params[4] * T_4(z)
              + params[5] * T_5(z)
              + params[6] * T_6(z)
              + params[7] * T_7(z)
              + params[8] * T_8(z)
              + params[9] * T_9(z)
    
    where z = x/3.0 maps [-3,3] to [-1,1].
    
    Chebyshev polynomials provide well-conditioned basis functions that
    help BFGS converge reliably from the all-ones starting point.
    """
    z = x / 3.0
    N = len(z)
    
    # Precompute Chebyshev polynomials T_0 through T_9
    basis = np.empty((N, 10))
    basis[:, 0] = 1.0
    basis[:, 1] = z
    
    for k in range(2, 10):
        basis[:, k] = 2.0 * z * basis[:, k-1] - basis[:, k-2]
    
    y = basis @ params
    return y
