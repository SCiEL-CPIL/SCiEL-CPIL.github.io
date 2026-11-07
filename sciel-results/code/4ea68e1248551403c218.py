"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers a function using a Chebyshev polynomial basis of total degree 3.
    
    The model is linear in the 10 parameters:
    f(x, y) = p0*T0(x)T0(y) + p1*T1(x)T0(y) + p2*T0(x)T1(y) +
              p3*T2(x)T0(y) + p4*T0(x)T2(y) + p5*T1(x)T1(y) +
              p6*T3(x)T0(y) + p7*T0(x)T3(y) + p8*T2(x)T1(y) +
              p9*T1(x)T2(y)
    
    Chebyshev polynomials are used for numerical stability and orthogonality
    on the domain [-1, 1], which aids BFGS optimization.
    """
    # Compute Chebyshev polynomials for x
    # T0 = 1
    # T1 = x
    # T2 = 2x^2 - 1
    # T3 = 4x^3 - 3x
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x**2 - 1.0
    T3_x = 4.0 * x**3 - 3.0 * x
    
    # Compute Chebyshev polynomials for y
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y**2 - 1.0
    T3_y = 4.0 * y**3 - 3.0 * y
    
    # Construct the 10 basis functions
    phi0 = T0_x * T0_y
    phi1 = T1_x * T0_y
    phi2 = T0_x * T1_y
    phi3 = T2_x * T0_y
    phi4 = T0_x * T2_y
    phi5 = T1_x * T1_y
    phi6 = T3_x * T0_y
    phi7 = T0_x * T3_y
    phi8 = T2_x * T1_y
    phi9 = T1_x * T2_y
    
    # Stack features into a matrix (N, 10)
    # Using np.stack with axis=1 is efficient
    Phi = np.stack((phi0, phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8, phi9), axis=1)
    
    # Compute predictions: Phi (N, 10) @ params (10,) -> (N,)
    pred = Phi @ params
    
    # Ensure finite values (safety)
    if not np.all(np.isfinite(pred)):
        # Fallback to simple linear if something went wrong
        return params[0] * x + params[1] * y + params[2]
        
    return pred
# EVOLVE-BLOCK-END
