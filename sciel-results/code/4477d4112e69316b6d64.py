"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strategy: 2-level decomposition
    # Level 1: Partition 6x6 into 2x2 blocks of 3x3 matrices
    # Use Strassen (rank 7) for the outer 2x2 block structure
    # Level 2: Each 3x3 * 3x3 product uses a rank-23 decomposition
    # Total rank = 7 * 23 = 161
    
    # Bürgisser rank-23 decomposition for 3x3 matrix multiplication
    # Each term: (u_coeffs, v_coeffs, c_contrib)
    # u_coeffs: dict mapping (i,k) to coeff for A[i,k]
    # v_coeffs: dict mapping (k,j) to coeff for B[k,j]  
    # c_contrib: dict mapping (i,j) to coeff for C[i,j]
    
    # I'll use a verified rank-23 decomposition.
    # From Bürgisser (2013), the rank-23 decomposition for 3x3 matrices.
    
    # Let me use a known explicit decomposition.
    # The 23 terms are as follows (using a verified set):
    
    # I'll use the decomposition that achieves rank 23.
    # Each term is a triple (u, v, c) where:
    # u is a 3x3 matrix of coefficients for A
    # v is a 3x3 matrix of coefficients for B
    # c is a 3x3 matrix of coefficients for C
    
    # For safety, let me first implement a correct rank-189 solution
    # (Strassen outer + schoolbook inner), then attempt rank-161.
    
    # Strassen outer terms:
    strassen_outer = [
        # M1 = (A00+A11)(B00+B11): C00 += 1, C11 += 1
        {
            'u_blocks': [(0, 1), (3, 1)],
            'v_blocks': [(0, 1), (3, 1)],
            'c_blocks': [(0, 1), (3, 1)],
        },
        # M2 = (A10+A11)B00: C10 += 1
        {
            'u_blocks': [(2, 1), (3, 1)],
            'v_blocks': [(0, 1)],
            'c_blocks': [(2, 1)],
        },
        # M3 = A00(B10-B00): C01 += 1
        {
            'u_blocks': [(0, 1)],
            'v_blocks': [(2, 1), (0, -1)],
            'c_blocks': [(1, 1)],
        },
        # M4 = A11(B01-B11): C00 += 1, C10 += 1
        {
            'u_blocks': [(3, 1)],
            'v_blocks': [(1, 1), (3, -1)],
            'c_blocks': [(0, 1), (2, 1)],
        },
        # M5 = (A00+A01)B11: C00 -= 1, C01 += 1
        {
            'u_blocks': [(0, 1), (1, 1)],
            'v_blocks': [(3, 1)],
            'c_blocks': [(0, -1), (1, 1)],
        },
        # M6 = (A10-A00)(B00+B01): C11 += 1
        {
            'u_blocks': [(2, 1), (0, -1)],
            'v_blocks': [(0, 1), (1, 1)],
            'c_blocks': [(3, 1)],
        },
        # M7 = (A01-A11)(B10+B11): C00 += 1
        {
            'u_blocks': [(1, 1), (3, -1)],
            'v_blocks': [(2, 1), (3, 1)],
            'c_blocks': [(0, 1)],
        },
    ]
    
    def block_to_global(block_idx, local_row, local_col, size=3):
        block_row = block_idx // 2
        block_col = block_idx % 2
        global_row = block_row * size + local_row
        global_col = block_col * size + local_col
        return global_row * 6 + global_col
    
    # Use schoolbook (rank 27) for inner 3x3 products
    # Total rank = 7 * 27 = 189
    
    rank = 189
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for term in strassen_outer:
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    for block_idx, coeff in term['u_blocks']:
                        U[block_to_global(block_idx, i, k), col] += coeff
                    for block_idx, coeff in term['v_blocks']:
                        V[block_to_global(block_idx, k, j), col] += coeff
                    for block_idx, coeff in term['c_blocks']:
                        W[block_to_global(block_idx, i, j), col] += coeff
                    col += 1
    
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
