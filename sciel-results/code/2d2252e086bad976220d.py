import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    # Structured Decomposition
    # Total rank 50
    R = 50
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_off, j_off, k_off):
        """
        Adds a Strassen M_{2,2,2} block.
        i_off, j_off, k_off are offsets in the global indices.
        Block A is A[i_off:i_off+2, j_off:j_off+2]
        Block B is B[j_off:j_off+2, k_off:k_off+2]
        Block C is C[i_off:i_off+2, k_off:k_off+2]
        """
        nonlocal col
        # Strassen intermediates M1..M7
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Coefficients for U (A), V (B), W (C)
        # Indices in local 2x2: 00, 01, 10, 11
        # Global U index: (i_off + li) * m + (j_off + lj)
        # Global V index: (j_off + lj) * p + (k_off + lk)
        # Global W index: (k_off + lk) * n + (i_off + li)
        
        blocks = [
            # (M_id, U_coeffs, V_coeffs, W_coeffs)
            # U_coeffs: dict (li, lj) -> coeff
            # V_coeffs: dict (lj, lk) -> coeff
            # W_coeffs: dict (lk, li) -> coeff
            {
                'u': {(0,0): 1, (1,1): 1},
                'v': {(0,0): 1, (1,1): 1},
                'w': {(0,0): 1, (1,1): 1, (0,1): -1, (1,0): 1} # C00+=M1, C11+=M1, C01-=M1, C10+=M1
            },
            {
                'u': {(1,0): 1, (1,1): 1},
                'v': {(0,0): 1},
                'w': {(0,1): 1, (1,1): -1} # C01+=M2, C11-=M2
            },
            {
                'u': {(0,0): 1},
                'v': {(0,1): 1, (1,1): -1},
                'w': {(0,1): 1, (1,1): 1} # C01+=M3, C11+=M3
            },
            {
                'u': {(1,1): 1},
                'v': {(1,0): 1, (0,0): -1},
                'w': {(0,0): 1, (1,0): 1} # C00+=M4, C10+=M4
            },
            {
                'u': {(0,0): 1, (0,1): 1},
                'v': {(1,1): 1},
                'w': {(0,0): -1, (0,1): 1} # C00-=M5, C01+=M5
            },
            {
                'u': {(1,0): 1, (0,0): -1},
                'v': {(0,0): 1, (0,1): 1},
                'w': {(1,1): 1} # C11+=M6
            },
            {
                'u': {(0,1): 1, (1,1): -1},
                'v': {(1,0): 1, (1,1): 1},
                'w': {(0,0): 1} # C00+=M7
            }
        ]
        
        for b in blocks:
            for (li, lj), cu in b['u'].items():
                U[(i_off + li) * m + (j_off + lj)][col] += cu
            for (lj, lk), cv in b['v'].items():
                V[(j_off + lj) * p + (k_off + lk)][col] += cv
            for (lk, li), cw in b['w'].items():
                W[(k_off + lk) * n + (i_off + li)][col] += cw
            col += 1

    # Add 6 Strassen blocks
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 0, 0)
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, 2, 0)
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 0, 2)
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_block(0, 2, 2)
    # Block 5: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    add_strassen_block(0, 0, 4)
    # Block 6: A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    add_strassen_block(0, 2, 4)
    
    # Add Schoolbook M_{2,4,1} block for k=6
    # C[i, 6] = sum_j A[i, j] B[j, 6]
    for i in range(2):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 6][col] = 1
            W[6 * n + i][col] = 1
            col += 1
            
    # Verify col == R
    if col != R:
        # Fallback if logic error
        return {"U": best_U, "V": best_V, "W": best_W}
        
    return {"U": U, "V": V, "W": W}

