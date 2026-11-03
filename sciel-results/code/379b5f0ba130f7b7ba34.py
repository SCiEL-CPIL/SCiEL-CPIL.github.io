import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    
    # We will build the decomposition explicitly.
    # Rank 54.
    # U shape: (10, 54)
    # V shape: (30, 54)
    # W shape: (12, 54)
    
    rank = 54
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    current_rank = 0
    
    def add_strassen_block(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Adds a Strassen 2x2x2 decomposition.
        A is 2x2 at a_row_offset, a_col_offset in A (2x5).
        B is 2x2 at b_row_offset, b_col_offset in B (5x6).
        C is 2x2 at c_row_offset, c_col_offset in C (2x6).
        
        Indices in U, V, W:
        U index: (i * m + j)
        V index: (j * p + k)
        W index: (k * n + i)
        """
        nonlocal current_rank
        
        # Strassen products
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = A21 (B11 + B22)
        # M3 = A11 (B22 - B11)
        # M4 = A22 (B11 - B21)
        # M5 = (A11 + A12) B22
        # M6 = (A21 - A11) (B11 + B12)
        # M7 = (A12 - A22) (B21 + B22)
        
        # A indices:
        # A11: (a_row_offset, a_col_offset)
        # A12: (a_row_offset, a_col_offset + 1)
        # A21: (a_row_offset + 1, a_col_offset)
        # A22: (a_row_offset + 1, a_col_offset + 1)
        
        # B indices:
        # B11: (b_row_offset, b_col_offset)
        # B12: (b_row_offset, b_col_offset + 1)
        # B21: (b_row_offset + 1, b_col_offset)
        # B22: (b_row_offset + 1, b_col_offset + 1)
        
        # C indices:
        # C11: (c_row_offset, c_col_offset)
        # C12: (c_row_offset, c_col_offset + 1)
        # C21: (c_row_offset + 1, c_col_offset)
        # C22: (c_row_offset + 1, c_col_offset + 1)
        
        # W index for C11: (c_col_offset * n + c_row_offset)
        # W index for C12: ((c_col_offset + 1) * n + c_row_offset)
        # W index for C21: (c_col_offset * n + (c_row_offset + 1))
        # W index for C22: ((c_col_offset + 1) * n + (c_row_offset + 1))
        
        w_c11 = c_col_offset * n + c_row_offset
        w_c12 = (c_col_offset + 1) * n + c_row_offset
        w_c21 = c_col_offset * n + (c_row_offset + 1)
        w_c22 = (c_col_offset + 1) * n + (c_row_offset + 1)
        
        u_a11 = a_row_offset * m + a_col_offset
        u_a12 = a_row_offset * m + (a_col_offset + 1)
        u_a21 = (a_row_offset + 1) * m + a_col_offset
        u_a22 = (a_row_offset + 1) * m + (a_col_offset + 1)
        
        v_b11 = b_row_offset * p + b_col_offset
        v_b12 = b_row_offset * p + (b_col_offset + 1)
        v_b21 = (b_row_offset + 1) * p + b_col_offset
        v_b22 = (b_row_offset + 1) * p + (b_col_offset + 1)
        
        # M1
        idx = current_rank; current_rank += 1
        U[u_a11, idx] += 1
        U[u_a22, idx] += 1
        V[v_b11, idx] += 1
        V[v_b22, idx] += 1
        W[w_c11, idx] += 1
        W[w_c22, idx] += 1
        
        # M2
        idx = current_rank; current_rank += 1
        U[u_a21, idx] += 1
        V[v_b11, idx] += 1
        V[v_b22, idx] += 1
        W[w_c21, idx] += 1
        W[w_c22, idx] -= 1
        
        # M3
        idx = current_rank; current_rank += 1
        U[u_a11, idx] += 1
        V[v_b22, idx] += 1
        V[v_b11, idx] -= 1
        W[w_c12, idx] += 1
        W[w_c22, idx] += 1
        
        # M4
        idx = current_rank; current_rank += 1
        U[u_a22, idx] += 1
        V[v_b11, idx] += 1
        V[v_b21, idx] -= 1
        W[w_c11, idx] += 1
        W[w_c21, idx] += 1
        
        # M5
        idx = current_rank; current_rank += 1
        U[u_a11, idx] += 1
        U[u_a12, idx] += 1
        V[v_b22, idx] += 1
        W[w_c11, idx] -= 1
        W[w_c12, idx] += 1
        
        # M6
        idx = current_rank; current_rank += 1
        U[u_a21, idx] += 1
        U[u_a11, idx] -= 1
        V[v_b11, idx] += 1
        V[v_b12, idx] += 1
        W[w_c22, idx] += 1
        
        # M7
        idx = current_rank; current_rank += 1
        U[u_a12, idx] += 1
        U[u_a22, idx] -= 1
        V[v_b21, idx] += 1
        V[v_b22, idx] += 1
        W[w_c11, idx] += 1
        W[w_c12, idx] -= 1

    def add_scalar_block(a_row, a_col, b_row, b_col, c_row, c_col):
        """
        Adds a scalar multiplication A[a_row, a_col] * B[b_row, b_col] to C[c_row, c_col].
        """
        nonlocal current_rank
        idx = current_rank; current_rank += 1
        U[a_row * m + a_col, idx] += 1
        V[b_row * p + b_col, idx] += 1
        W[c_col * n + c_row, idx] += 1

    # Part 1a: 2x4x4
    # A cols 0-3, B rows 0-3, B cols 0-3
    # Blocks:
    # A_L (cols 0,1), A_R (cols 2,3)
    # B_LL (rows 0,1; cols 0,1), B_LR (rows 0,1; cols 2,3)
    # B_RL (rows 2,3; cols 0,1), B_RR (rows 2,3; cols 2,3)
    
    # A_L B_LL
    add_strassen_block(0, 0, 0, 0, 0, 0) # C cols 0,1
    # A_L B_LR
    add_strassen_block(0, 0, 0, 2, 0, 2) # C cols 2,3
    # A_R B_RL
    add_strassen_block(0, 2, 2, 0, 0, 0) # C cols 0,1
    # A_R B_RR
    add_strassen_block(0, 2, 2, 2, 0, 2) # C cols 2,3
    
    # Part 1b: 2x4x2
    # A cols 0-3, B rows 0-3, B cols 4,5
    # A_L (cols 0,1), A_R (cols 2,3)
    # B_L (rows 0,1; cols 4,5), B_R (rows 2,3; cols 4,5)
    
    # A_L B_L
    add_strassen_block(0, 0, 0, 4, 0, 4) # C cols 4,5
    # A_R B_R
    add_strassen_block(0, 2, 2, 4, 0, 4) # C cols 4,5
    
    # Part 2: 2x1x6
    # A col 4, B row 4, B cols 0-5
    for i in range(2):
        for k in range(6):
            add_scalar_block(i, 4, 4, k, i, k)
            
    # Verify
    # We expect current_rank to be 54.
    if current_rank != 54:
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
