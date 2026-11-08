"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # Strategy: Split inner dimension m=3 into m=1+2.
    # Part 1: 2x1x6 (A row vector * B column vector) -> rank 12
    # Part 2: 2x2x6 (Strassen 2x2x2 x 2 blocks) -> rank 14
    # Total rank = 26
    rank = 26
    
    U = [[0] * rank for _ in range(n * m)]   # 6 x 26
    V = [[0] * rank for _ in range(m * p)]   # 18 x 26
    W = [[0] * rank for _ in range(p * n)]   # 12 x 26
    
    # === PART 1: 2x1x6 (rank 12) ===
    # A[0,2] * B[2,k] -> C[0,k]
    # A[1,2] * B[2,k] -> C[1,k]
    # U index: i*m + j -> A[i,2] at i*3+2
    # V index: j*p + k -> B[2,k] at 2*6+k
    # W index: k*n + i -> C[i,k] at k*2+i
    for k in range(p):
        for i in range(n):
            col = k * n + i
            U[i * m + 2][col] = 1       # A[i, 2]
            V[2 * p + k][col] = 1       # B[2, k]
            W[k * n + i][col] = 1       # C[i, k]
    
    # === PART 2: 2x2x6 (rank 14) ===
    # Uses A[:, 0:2] (2x2) and B[0:2, :] (2x6)
    # Strassen 7 terms for 2x2x2, repeated for 2 column-blocks of size 2
    #
    # Strassen terms (standard):
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    # M2 = A11(B10-B00)       -> C10 += M2
    # M3 = (A10-A00)(B00+B01) -> C11 -= M3
    # M4 = A00(B01-B11)       -> C01 += M4
    # M5 = (A01+A11)(B10+B11) -> C01 += M5
    # M6 = (A10+A11)(B01-B10) -> C10 += M6
    # M7 = (A00-A10)(B10+B11) -> C00 -= M7
    #
    # A[i,j] local: i in {0,1}, j in {0,1}, global U index = i*3+j
    # B[j,k] local: j in {0,1}, k in {0,1}, global V index = j*6 + block*2 + k
    # C[i,k] local: i in {0,1}, k in {0,1}, global W index = (block*2+k)*2 + i
    
    strassen = [
        # (u: {local_A_idx: coeff}, v: {local_B_idx: coeff}, w: {local_C_idx: coeff})
        # local_A_idx = i*2+j, local_B_idx = j*2+k, local_C_idx = i*2+k
        # M1 = (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = A11*(B10-B00)
        ({3: 1}, {2: 1, 0: -1}, {2: 1}),
        # M3 = (A10-A00)*(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: -1}),
        # M4 = A00*(B01-B11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1}),
        # M5 = (A01+A11)*(B10+B11)
        ({1: 1, 3: 1}, {2: 1, 3: 1}, {1: 1}),
        # M6 = (A10+A11)*(B01-B10)
        ({2: 1, 3: 1}, {1: 1, 2: -1}, {2: 1}),
        # M7 = (A00-A10)*(B10+B11)
        ({0: 1, 2: -1}, {2: 1, 3: 1}, {0: -1}),
    ]
    
    # Apply Strassen terms for each of the 3 column-blocks of size 2
    for block in range(3):
        for s_idx, (u_dict, v_dict, w_dict) in enumerate(strassen):
            col = 12 + block * 7 + s_idx  # offset by 12 for part 1
            
            # Fill U: local A index = i*2+j -> global U index = i*3+j
            for la, val in u_dict.items():
                i_local = la // 2
                j_local = la % 2
                U[i_local * m + j_local][col] = val
            
            # Fill V: local B index = j*2+k -> global V index = j*6 + (block*2+k)
            for lb, val in v_dict.items():
                j_local = lb // 2
                k_local = lb % 2
                V[j_local * p + block * 2 + k_local][col] = val
            
            # Fill W: local C index = i*2+k -> global W index = (block*2+k)*2 + i
            for lc, val in w_dict.items():
                i_local = lc // 2
                k_local = lc % 2
                W[(block * 2 + k_local) * n + i_local][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
