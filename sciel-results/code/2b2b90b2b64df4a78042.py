"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes the prediction using a separable tensor-product Chebyshev basis.
    
    Basis: T_i(x) * T_j(y) for i, j in {0, 1, 2}, giving 9 terms.
    Plus one additional term T_3(x) * T_3(y) to capture degree-3 interactions.
    
    Chebyshev polynomials on [-1, 1]:
    T_0(t) = 1
    T_1(t) = t
    T_2(t) = 2t^2 - 1
    T_3(t) = 4t^3 - 3t
    
    This basis is well-conditioned and captures smooth functions effectively.
    """
    # Compute Chebyshev polynomials for x and y
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x**2 - 1.0
    T3_x = 4.0 * x**3 - 3.0 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y**2 - 1.0
    T3_y = 4.0 * y**3 - 3.0 * y
    
    # Tensor product basis: T_i(x) * T_j(y) for i,j in {0,1,2}
    # That's 9 terms, indexed 0-8
    pred = np.zeros_like(x, dtype=np.float64)
    
    idx = 0
    for Tx in [T0_x, T1_x, T2_x]:
        for Ty in [T0_y, T1_y, T2_y]:
            pred += params[idx] * Tx * Ty
            idx += 1
    
    # 10th term: T_3(x) * T_3(y) for higher-order interaction
    pred += params[9] * T3_x * T3_y
    
    return pred
# EVOLVE-BLOCK-END
