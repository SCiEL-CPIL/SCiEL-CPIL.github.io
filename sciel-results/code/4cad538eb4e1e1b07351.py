"""Matrix multiplication 6x6x6 - Strassen on 2x2 grid of 3x3 blocks."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-structured decomposition for 6x6 matrix multiplication.
    Partition 6x6 into 2x2 grid of 3x3 blocks.
    Apply Strassen (rank 7) to the 2x2 block matrix multiplication.
    For each Strassen term, the "scalar" products are 3x3 matrix multiplications (rank 9 schoolbook).
    Total rank = 7 * 9 = 63.
    
    W follows convention W[k*n+i, r] where (k,i) is (col_C, row_C).
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    bs = 3       # block size
    nb = 2       # number of blocks per dimension
    
    total_rank = 7 * 9  # 63
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2
    # A entries: 0=A00, 1=A01, 2=A10, 3=A11
    # B entries: 0=B00, 1=B01, 2=B10, 3=B11
    # C entries: 0=C00, 1=C01, 2=C10, 3=C11
    
    # Correct Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A00-A11)*(B00-B11)
    # M3 = (A10+A11)*B00
    # M4 = (A00+A01)*B10
    # M5 = (A01-A00)*(B11-B00)
    # M6 = (A10-A11)*(B11+B00)
    # M7 = (A11-A10)*(B01-B10)
    #
    # C00 = M1 + M2
    # C01 = M1 + M5
    # C10 = M3 + M6
    # C11 = M4 + M2 + M7
    
    strassen_terms = [
        # (a_coeffs[4], b_coeffs[4], c_coeffs[4])
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 0, 0]),  # M1
        ([1, 0, 0, -1], [1, 0, 0, -1], [1, 0, 0, 1]),  # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),  # M3
        ([1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]),  # M4
        ([-1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),  # M5
        ([0, 0, 1, -1], [1, 0, 0, 1], [0, 0, 1, 0]),  # M6
        ([0, 0, -1, 1], [0, 1, -1, 0], [0, 0, 0, 1]),  # M7
    ]
    
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    for t_idx in range(7):
        a_coeffs, b_coeffs, c_coeffs = strassen_terms[t_idx]
        
        # For each schoolbook product in 3x3:
        for i in range(bs):
            for k in range(bs):
                for j in range(bs):
                    # U: for each block a, add a_coeffs[a] to U[row, col]
                    for a in range(4):
                        if a_coeffs[a] != 0:
                            a_row = (a // 2) * bs + i
                            a_col = (a % 2) * bs + k
                            u_idx = a_row * m + a_col
                            U[u_idx][col] += a_coeffs[a]
                    
                    # V: for each block b, add b_coeffs[b] to V[row, col]
                    for b in range(4):
                        if b_coeffs[b] != 0:
                            b_row = (b // 2) * bs + k
                            b_col = (b % 2) * bs + j
                            v_idx = b_row * p + b_col
                            V[v_idx][col] += b_coeffs[b]
                    
                    # W: for each block c, add c_coeffs[c] to W[row, col]
                    for c in range(4):
                        if c_coeffs[c] != 0:
                            c_row = (c // 2) * bs + i
                            c_col = (c % 2) * bs + j
                            w_idx = c_col * n + c_row
                            W[w_idx][col] += c_coeffs[c]
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
