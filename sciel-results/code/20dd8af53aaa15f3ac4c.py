"""Matrix multiplication 2x4 x 4x8 - rank minimal decomposition."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Construct a rank-minimal decomposition for 2x4 * 4x8 matrix multiplication."""
    n, m, p = 2, 4, 8  # A is n x m, B is m x p, C is n x p
    
    # Rank-56 decomposition using Strassen's algorithm on 2x2 blocks
    # We split into 4 column-blocks of size 2, each requiring 7 multiplications
    # for the 2x2 * 2x2 products, giving 4 * 7 = 28 multiplications.
    # But we also need to handle the row structure of B (4 rows = 2 blocks of 2).
    
    # Strategy: For each output column block (2 columns), we compute:
    # C[:, 2k:2k+2] = A[:, 0:2] @ B[0:2, 2k:2k+2] + A[:, 2:4] @ B[2:4, 2k:2k+2]
    # Each 2x2 * 2x2 multiplication uses 7 Strassen terms.
    # Total: 4 column-blocks * 2 row-blocks * 7 terms = 56 multiplications.
    
    # Strassen terms for 2x2 matrix multiplication
    # C = A * B where A, B, C are 2x2
    # Using local indices (0,0), (0,1), (1,0), (1,1)
    
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # M1 = (a00+a11)(b00+b11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (a10+a11)(b00)
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = (a00)(b01-b11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = (a11)(b10-b00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (a00+a01)(b11)
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (a10-a00)(b00+b01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (a01-a11)(b10+b11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # n*m = 8
    V = [[0] * rank for _ in range(m * p)]  # m*p = 32
    W = [[0] * rank for _ in range(p * n)]  # p*n = 16
    
    col = 0
    for k_block in range(4):  # 4 column blocks of size 2
        for j_block in range(2):  # 2 row blocks of B (rows 0-1, 2-3)
            for (A_coeffs, B_coeffs, C_coeffs) in strassen_terms:
                # Fill U: A coefficients
                # A is 2x4, we use columns j_block*2 and j_block*2+1
                for (li, lj), coeff in A_coeffs.items():
                    global_i = li
                    global_j = j_block * 2 + lj
                    U[global_i * m + global_j][col] += coeff
                
                # Fill V: B coefficients
                # B is 4x8, we use rows j_block*2 and j_block*2+1, columns k_block*2 and k_block*2+1
                for (lj, lk), coeff in B_coeffs.items():
                    global_j = j_block * 2 + lj
                    global_k = k_block * 2 + lk
                    V[global_j * p + global_k][col] += coeff
                
                # Fill W: C coefficients
                # C is 2x8, we use rows 0 and 1, columns k_block*2 and k_block*2+1
                for (li, lk), coeff in C_coeffs.items():
                    global_i = li
                    global_k = k_block * 2 + lk
                    W[global_k * n + global_i][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
