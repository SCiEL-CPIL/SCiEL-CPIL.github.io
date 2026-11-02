import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank-42 Strassen-based decomposition:
    # Partition 12 columns into 6 blocks of 2, apply 7-product Strassen to each pair.
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 7 terms for 2x2 * 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # U terms: coefficients on A[i,j] for each product
    # V terms: coefficients on B[j,k] for each product
    # W terms: coefficients on C[i,k] (which is output[i,k]) for each product
    
    terms = [
        # M1 = (A00+A11)*(B00+B11) -> C00 +=1, C11 +=1
        {'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}},
        # M2 = (A10+A11)*B00 -> C10 +=1, C11 +=-1
        {'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(0,1): -1, (1,0): 1}},
        # M3 = A00*(B01-B11) -> C01 +=1, C11 +=1
        {'U': {(0,0): 1}, 'V': {(0,1): 1, (1,1): -1}, 'W': {(0,1): 1, (1,1): 1}},
        # M4 = A11*(B10-B00) -> C00 +=1, C10 +=1
        {'U': {(1,1): 1}, 'V': {(1,0): 1, (0,0): -1}, 'W': {(0,0): 1, (1,0): 1}},
        # M5 = (A00+A01)*B11 -> C00 +=-1, C01 +=1
        {'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,0): -1, (0,1): 1}},
        # M6 = (A10-A00)*(B00+B01) -> C11 +=1
        {'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(1,1): 1}},
        # M7 = (A01-A11)*(B10+B11) -> C00 +=1
        {'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1}}
    ]
    
    num_blocks = p // 2
    for b in range(num_blocks):
        for t_idx in range(7):
            r = b * 7 + t_idx
            term = terms[t_idx]
            for (i, j), val in term['U'].items():
                U[i * m + j][r] = val
            for (j, k_loc), val in term['V'].items():
                k_glob = b * 2 + k_loc
                V[j * p + k_glob][r] = val
            for (k_loc, i), val in term['W'].items():
                k_glob = b * 2 + k_loc
                W[k_glob * n + i][r] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
