import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x8 matrix multiplication.
    Uses rank-28 Strassen solution.
    """
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # Known rank-28 solution
    return _strassen_28()

def _strassen_28():
    """Returns rank-28 solution using Strassen per 2x2 block."""
    n, m, p = DIMENSIONS
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    A_forms = [
        {0: 1, 3: 1},       # M1: a+d
        {2: 1, 3: 1},       # M2: c+d
        {0: 1},             # M3: a
        {3: 1},             # M4: d
        {0: 1, 1: 1},       # M5: a+b
        {2: 1, 0: -1},      # M6: c-a
        {1: 1, 3: -1}       # M7: b-d
    ]
    
    B_forms = [
        {0: 1, 3: 1},       # M1: x0+y1
        {0: 1},             # M2: x0
        {1: 1, 3: -1},      # M3: x1-y1
        {2: 1, 0: -1},      # M4: y0-x0
        {3: 1},             # M5: y1
        {0: 1, 1: 1},       # M6: x0+x1
        {2: 1, 3: 1}        # M7: y0+y1
    ]
    
    C_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1}, # M1, M4, M5, M7
        {2: 1, 4: 1},              # M3, M5
        {1: 1, 3: 1},              # M2, M4
        {0: 1, 1: -1, 2: 1, 5: 1}  # M1, M2, M3, M6
    ]

    for block in range(4):
        k0 = block * 2
        
        for s in range(7):
            r = block * 7 + s
            
            for idx, val in A_forms[s].items():
                i = idx // 2
                j = idx % 2
                U[i * m + j][r] = val
            
            for idx, val in B_forms[s].items():
                j = idx // 2
                k_local = idx % 2
                k_global = k0 + k_local
                V[j * p + k_global][r] = val
            
            for c_idx, coeffs in enumerate(C_coeffs):
                if s in coeffs:
                    coeff = coeffs[s]
                    i = c_idx // 2
                    k_local = c_idx % 2
                    k_global = k0 + k_local
                    W[k_global * n + i][r] = coeff

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
