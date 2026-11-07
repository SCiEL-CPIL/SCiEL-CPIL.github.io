"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a cubic polynomial basis in x and y.
    
    The model is:
    z = p[0] + p[1]*x + p[2]*y 
      + p[3]*x^2 + p[4]*x*y + p[5]*y^2 
      + p[6]*x^3 + p[7]*x^2*y + p[8]*x*y^2 + p[9]*y^3
    
    This utilizes all 10 parameters to span the space of degree-3 polynomials.
    """
    p = params
    
    # Ensure parameters are treated as floats to avoid integer division issues if params are passed as ints
    p = p.astype(float)
    
    # Compute Chebyshev basis terms
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    xy = x * y
    
    # Chebyshev polynomials for x
    Tx0 = np.ones_like(x)
    Tx1 = x
    Tx2 = 2 * x2 - 1
    Tx3 = 4 * x3 - 3 * x
    
    # Chebyshev polynomials for y
    Ty0 = np.ones_like(y)
    Ty1 = y
    Ty2 = 2 * y2 - 1
    Ty3 = 4 * y3 - 3 * y
    
    # Construct the linear combination using tensor products of Chebyshev polynomials
    # This spans the same space as full cubic polynomials but is better conditioned.
    # p[0]: T0(x)T0(y) = 1
    # p[1]: T1(x)T0(y) = x
    # p[2]: T0(x)T1(y) = y
    # p[3]: T2(x)T0(y) = 2x^2 - 1
    # p[4]: T1(x)T1(y) = xy
    # p[5]: T0(x)T2(y) = 2y^2 - 1
    # p[6]: T3(x)T0(y) = 4x^3 - 3x
    # p[7]: T2(x)T1(y) = (2x^2 - 1)y
    # p[8]: T1(x)T2(y) = x(2y^2 - 1)
    # p[9]: T0(x)T3(y) = 4y^3 - 3y
    
    result = (p[0] * Tx0 * Ty0 
              + p[1] * Tx1 * Ty0 
              + p[2] * Tx0 * Ty1 
              + p[3] * Tx2 * Ty0 
              + p[4] * Tx1 * Ty1 
              + p[5] * Tx0 * Ty2 
              + p[6] * Tx3 * Ty0 
              + p[7] * Tx2 * Ty1 
              + p[8] * Tx1 * Ty2 
              + p[9] * Tx0 * Ty3)
              
    return result
# EVOLVE-BLOCK-END
