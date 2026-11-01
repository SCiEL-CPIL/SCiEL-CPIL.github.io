"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-21 Strassen decomposition for M_{2,2,6}.
    """
    n, m, p = DIMENSIONS  # 2, 2, 6
    
    # Build the rank-21 Strassen decomposition
    return _build_rank21_strassen(n, m, p)

def _build_rank21_strassen(n, m, p):
    """Build the rank-21 Strassen decomposition."""
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # u_coeffs: {A_idx: val}, v_coeffs: {B_idx: val}, w_coeffs: {(C_i, C_k): val}
    # A indices: 0,1,2,3 corresponding to A00, A01, A10, A11
    # B indices: 0,1,2,3 corresponding to B00, B01, B10, B11
    # C indices: (i, k) corresponding to C_{i,k}
    
    strassen = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]
    
    r = 0
    for b in range(3):
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen[t]
            
            for idx, val in u_coeffs.items():
                U[idx][r] = val
            
            for l, val in v_coeffs.items():
                # Map local B index (0-3) to global B index (0-11)
                # l=0 -> B[0, 2b], l=1 -> B[0, 2b+1]
                # l=2 -> B[1, 2b], l=3 -> B[1, 2b+1]
                row = l // 2
                col = l % 2
                v_idx = row * p + (2 * b + col)
                V[v_idx][r] = val
            
            for (i, k_local), val in w_coeffs.items():
                # Map local C index (i, k_local) to global C index
                # k_global = 2b + k_local
                # C index in W is k_global * n + i
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx][r] = val
            
            r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
