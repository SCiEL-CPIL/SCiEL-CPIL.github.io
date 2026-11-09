import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    
    # Legendre polynomials
    L0_x = np.ones_like(x)
    L1_x = x
    L2_x = 0.5 * (3.0 * x * x - 1.0)
    L3_x = 0.5 * (5.0 * x * x * x - 3.0 * x)
    
    L0_y = np.ones_like(y)
    L1_y = y
    L2_y = 0.5 * (3.0 * y * y - 1.0)
    L3_y = 0.5 * (5.0 * y * y * y - 3.0 * y)
    
    # Normalization factors to make basis functions approximately unit norm
    # on [-1, 1] x [-1, 1]
    # Norm(L_n) = sqrt(2/(2n+1))
    # Norm(L_i(x)L_j(y)) = Norm(L_i) * Norm(L_j)
    # We want to divide by this norm to get unit norm basis.
    
    n0 = np.sqrt(2.0)       # Norm L0
    n1 = np.sqrt(2.0/3.0)   # Norm L1
    n2 = np.sqrt(2.0/5.0)   # Norm L2
    n3 = np.sqrt(2.0/7.0)   # Norm L3
    
    # Term 0: L0_x * L0_y. Norm = n0 * n0 = 2. Scale = 1/2
    term0 = (L0_x * L0_y) / (n0 * n0)
    
    # Term 1: L1_x * L0_y. Norm = n1 * n0. Scale = 1/(n1*n0)
    term1 = (L1_x * L0_y) / (n1 * n0)
    
    # Term 2: L0_x * L1_y. Norm = n0 * n1. Scale = 1/(n0*n1)
    term2 = (L0_x * L1_y) / (n0 * n1)
    
    # Term 3: L2_x * L0_y. Norm = n2 * n0. Scale = 1/(n2*n0)
    term3 = (L2_x * L0_y) / (n2 * n0)
    
    # Term 4: L0_x * L2_y. Norm = n0 * n2. Scale = 1/(n0*n2)
    term4 = (L0_x * L2_y) / (n0 * n2)
    
    # Term 5: L1_x * L1_y. Norm = n1 * n1 = 2/3. Scale = 3/2
    term5 = (L1_x * L1_y) / (n1 * n1)
    
    # Term 6: L3_x * L0_y. Norm = n3 * n0. Scale = 1/(n3*n0)
    term6 = (L3_x * L0_y) / (n3 * n0)
    
    # Term 7: L0_x * L3_y. Norm = n0 * n3. Scale = 1/(n0*n3)
    term7 = (L0_x * L3_y) / (n0 * n3)
    
    # Term 8: L2_x * L1_y. Norm = n2 * n1. Scale = 1/(n2*n1)
    term8 = (L2_x * L1_y) / (n2 * n1)
    
    # Term 9: L1_x * L2_y. Norm = n1 * n2. Scale = 1/(n1*n2)
    term9 = (L1_x * L2_y) / (n1 * n2)

    return (p[0] * term0 +
            p[1] * term1 +
            p[2] * term2 +
            p[3] * term3 +
            p[4] * term4 +
            p[5] * term5 +
            p[6] * term6 +
            p[7] * term7 +
            p[8] * term8 +
            p[9] * term9)
