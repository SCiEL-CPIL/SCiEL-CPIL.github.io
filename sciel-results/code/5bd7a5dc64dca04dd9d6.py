"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-189 decomposition of the 6x6x6 matrix multiplication tensor
    using Strassen's algorithm applied to 2x2 blocks arranged in a 3x3 grid.
    
    The 6x6 matrices are partitioned into nine 2x2 blocks. For each triple of blocks
    (I,J,K) corresponding to C[I,K] += A[I,J]*B[J,K], we apply Strassen's rank-7
    algorithm, contributing 7 scalar multiplications per block triple.
    
    Total rank = 3^3 * 7 = 189.
    """
    n, m, p = DIMENSIONS
    block_size = 2
    blocks = 3  # 6 / 2
    
    rank = (blocks ** 3) * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 product terms for 2x2 matrix multiplication.
    # Each term is (u_pattern, v_pattern, w_pattern) where each pattern maps
    # (local_row, local_col) -> coefficient.
    #
    # The 7 multiplications are:
    #   M1 = (A11+A22)(B11+B22)
    #   M2 = (A21+A22)B11
    #   M3 = A11(B21-B22)
    #   M4 = A22(B11-B21)
    #   M5 = (A11+A12)B22
    #   M6 = (A21-A11)(B11+B12)
    #   M7 = (A12-A22)(B21+B22)
    #
    # Output: C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    
    # For each term, define the linear combinations of A, B, and C entries.
    # u_pattern: coefficients for A entries (row, col) in the 2x2 block
    # v_pattern: coefficients for B entries (row, col) in the 2x2 block
    # w_pattern: coefficients for C entries (row, col) in the 2x2 block
    
    terms = [
        # M1 = (A11+A22)(B11+B22)
        # U: A11 + A22, V: B11 + B22, W: C11 + C22
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A21+A22)B11
        # U: A21 + A22, V: B11, W: C21 - C22
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A11(B21-B22)
        # U: A11, V: B21 - B22, W: C12 + C22
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A22(B11-B21)
        # U: A22, V: B11 - B21, W: C11 + C21
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A11+A12)B22
        # U: A11 + A12, V: B22, W: -C11 + C12
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A21-A11)(B11+B12)
        # U: A21 - A11, V: B11 + B12, W: C22
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A12-A22)(B21+B22)
        # U: A12 - A22, V: B21 + B22, W: C11
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    col = 0
    for I in range(blocks):
        for J in range(blocks):
            for K in range(blocks):
                for u_pat, v_pat, w_pat in terms:
                    # Fill U: maps (n*m) row index to coefficient
                    for (lr, lc), coeff in u_pat.items():
                        row = I * block_size + lr
                        col_a = J * block_size + lc
                        U[row * m + col_a][col] += coeff
                    # Fill V: maps (m*p) row index to coefficient
                    for (lr, lc), coeff in v_pat.items():
                        row = J * block_size + lr
                        col_b = K * block_size + lc
                        V[row * p + col_b][col] += coeff
                    # Fill W: maps (p*n) row index to coefficient
                    for (lr, lc), coeff in w_pat.items():
                        row = K * block_size + lr
                        col_c = I * block_size + lc
                        W[row * n + col_c][col] += coeff
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
