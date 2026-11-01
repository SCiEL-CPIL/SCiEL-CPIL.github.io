"""Matrix multiplication 2x2x14: rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-49 exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x14 matrix using Strassen's algorithm.
    
    For 2x2 * 2x14, we can view the 2x14 matrix as 7 independent 2x2 blocks
    (since 14 = 7*2). For each 2x2 block, we use Strassen's algorithm which
    gives rank 7 instead of rank 8. Total rank = 7 * 7 = 49.
    
    Strassen's algorithm for 2x2x2:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B10)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1+M4-M5+M7
    C01 = M3+M5
    C10 = M2+M4
    C11 = M1-M2+M3+M6
    
    For 2x2 * 2x14, we split the 14 columns into 7 pairs. For each pair of
    columns, we apply Strassen's algorithm. The 2x2 matrix A is shared across
    all 7 blocks, but the B blocks are independent.
    
    In the tensor decomposition:
    - U has shape (4, 49) - corresponds to the 4 entries of A
    - V has shape (28, 49) - corresponds to the 28 entries of B (2 rows x 14 cols)
    - W has shape (28, 49) - corresponds to the 28 entries of C (2 rows x 14 cols)
    
    For each Strassen term in each block, we have one rank-1 term.
    """
    n, m, p = DIMENSIONS  # 2, 2, 14
    
    # Number of 2x2 blocks in the 14 columns
    num_blocks = p // 2  # 7 blocks
    
    # Strassen coefficients for 2x2x2
    # Each entry: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: 4 values for A entries [A00, A01, A10, A11]
    # v_coeffs: 4 values for B entries [B00, B01, B10, B11]
    # w_coeffs: 4 values for C entries [C00, C01, C10, C11]
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2 = (A10+A11)B00
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3 = A00(B10-B11)
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),
        # M4 = A11(B01-B10)
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),
        # M5 = (A00+A01)B11
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6 = (A10-A00)(B00+B10)
        ([-1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]),
        # M7 = (A01-A11)(B01+B11)
        ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 0]),
    ]
    
    rank = 7 * num_blocks  # 49
    U = [[0] * rank for _ in range(n * m)]      # 4 x 49
    V = [[0] * rank for _ in range(m * p)]      # 28 x 49
    W = [[0] * rank for _ in range(p * n)]      # 28 x 49
    
    term_idx = 0
    for block in range(num_blocks):
        # Column range for this block: 2*block to 2*block+1
        col_start = 2 * block
        col_end = col_start + 2
        
        for (u_coeffs, v_coeffs, w_coeffs) in strassen_terms:
            # Set U coefficients (same for all blocks)
            for i in range(4):
                U[i][term_idx] = u_coeffs[i]
            
            # Set V coefficients
            # B indices: (j, k) where j in {0,1}, k in {0..13}
            # For this block, B entries are at positions:
            # B00 -> (0, col_start), B01 -> (0, col_start+1)
            # B10 -> (1, col_start), B11 -> (1, col_start+1)
            v_positions = [
                0 * p + col_start,      # B00
                0 * p + col_start + 1,  # B01
                1 * p + col_start,      # B10
                1 * p + col_start + 1,  # B11
            ]
            for i in range(4):
                V[v_positions[i]][term_idx] = v_coeffs[i]
            
            # Set W coefficients
            # C indices: (k, i) where k in {0..13}, i in {0,1}
            # Wait, the output is C[i,k] where i in {0,1}, k in {0..13}
            # The W matrix has shape (p*n, rank) = (28, rank)
            # Index convention: W[k*n+i, r]
            # So C[i,k] is at position k*n + i = k*2 + i
            # C entries for this block:
            # C00 -> (col_start, 0) -> position col_start*2 + 0
            # C01 -> (col_start+1, 0) -> position (col_start+1)*2 + 0
            # C10 -> (col_start, 1) -> position col_start*2 + 1
            # C11 -> (col_start+1, 1) -> position (col_start+1)*2 + 1
            w_positions = [
                col_start * n + 0,           # C00
                (col_start + 1) * n + 0,     # C01
                col_start * n + 1,           # C10
                (col_start + 1) * n + 1,     # C11
            ]
            for i in range(4):
                W[w_positions[i]][term_idx] = w_coeffs[i]
            
            term_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
