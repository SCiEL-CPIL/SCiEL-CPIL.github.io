"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    rank = 50
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    def add_strassen_block(a_i_offset, a_j_offset, b_j_offset, b_k_offset, start_col):
        # M1
        U[a_i_offset * m + a_j_offset + 0, start_col + 0] = 1
        U[a_i_offset * m + a_j_offset + 1, start_col + 0] = 1
        V[(b_j_offset + 0) * p + b_k_offset + 0, start_col + 0] = 1
        V[(b_j_offset + 1) * p + b_k_offset + 1, start_col + 0] = 1
        W[(b_k_offset + 0) * n + a_i_offset + 0, start_col + 0] = 1
        W[(b_k_offset + 1) * n + a_i_offset + 1, start_col + 0] = 1
        
        # M2
        U[a_i_offset * m + a_j_offset + 0, start_col + 1] = 1
        U[a_i_offset * m + a_j_offset + 1, start_col + 1] = 1
        V[(b_j_offset + 0) * p + b_k_offset + 0, start_col + 1] = 1
        W[(b_k_offset + 0) * n + a_i_offset + 1, start_col + 1] = 1
        
        # M3
        U[a_i_offset * m + a_j_offset + 0, start_col + 2] = 1
        V[(b_j_offset + 0) * p + b_k_offset + 1, start_col + 2] = 1
        V[(b_j_offset + 1) * p + b_k_offset + 1, start_col + 2] = -1
        W[(b_k_offset + 1) * n + a_i_offset + 0, start_col + 2] = 1
        
        # M4
        U[a_i_offset * m + a_j_offset + 1, start_col + 3] = 1
        V[(b_j_offset + 1) * p + b_k_offset + 0, start_col + 3] = 1
        V[(b_j_offset + 0) * p + b_k_offset + 0, start_col + 3] = -1
        W[(b_k_offset + 0) * n + a_i_offset + 0, start_col + 3] = 1
        W[(b_k_offset + 0) * n + a_i_offset + 1, start_col + 3] = 1
        
        # M5
        U[a_i_offset * m + a_j_offset + 0, start_col + 4] = 1
        U[a_i_offset * m + a_j_offset + 1, start_col + 4] = 1
        V[(b_j_offset + 1) * p + b_k_offset + 1, start_col + 4] = 1
        W[(b_k_offset + 0) * n + a_i_offset + 0, start_col + 4] = -1
        W[(b_k_offset + 1) * n + a_i_offset + 0, start_col + 4] = 1
        
        # M6
        U[a_i_offset * m + a_j_offset + 0, start_col + 5] = 1
        U[a_i_offset * m + a_j_offset + 1, start_col + 5] = -1
        V[(b_j_offset + 0) * p + b_k_offset + 0, start_col + 5] = 1
        V[(b_j_offset + 0) * p + b_k_offset + 1, start_col + 5] = 1
        W[(b_k_offset + 1) * n + a_i_offset + 1, start_col + 5] = 1
        
        # M7
        U[a_i_offset * m + a_j_offset + 0, start_col + 6] = 1
        U[a_i_offset * m + a_j_offset + 1, start_col + 6] = -1
        V[(b_j_offset + 1) * p + b_k_offset + 0, start_col + 6] = 1
        V[(b_j_offset + 1) * p + b_k_offset + 1, start_col + 6] = 1
        W[(b_k_offset + 0) * n + a_i_offset + 0, start_col + 6] = 1
        
        return start_col + 7

    def add_schoolbook_2x2x1_block(a_i_offset, a_j_offset, b_j_offset, b_k_offset, start_col):
        # Term 1: A[0,0] B[0,0] -> Z[0,0]
        U[a_i_offset * m + a_j_offset + 0, start_col + 0] = 1
        V[(b_j_offset + 0) * p + b_k_offset + 0, start_col + 0] = 1
        W[(b_k_offset + 0) * n + a_i_offset + 0, start_col + 0] = 1
        
        # Term 2: A[0,1] B[1,0] -> Z[0,0]
        U[a_i_offset * m + a_j_offset + 1, start_col + 1] = 1
        V[(b_j_offset + 1) * p + b_k_offset + 0, start_col + 1] = 1
        W[(b_k_offset + 0) * n + a_i_offset + 0, start_col + 1] = 1
        
        # Term 3: A[1,0] B[0,0] -> Z[1,0]
        U[a_i_offset * m + a_j_offset + 0, start_col + 2] = 1
        V[(b_j_offset + 0) * p + b_k_offset + 0, start_col + 2] = 1
        W[(b_k_offset + 0) * n + a_i_offset + 1, start_col + 2] = 1
        
        # Term 4: A[1,1] B[1,0] -> Z[1,0]
        U[a_i_offset * m + a_j_offset + 1, start_col + 3] = 1
        V[(b_j_offset + 1) * p + b_k_offset + 0, start_col + 3] = 1
        W[(b_k_offset + 0) * n + a_i_offset + 1, start_col + 3] = 1
        
        return start_col + 4

    # A1 (cols 0-1), B1 (rows 0-1)
    col = add_strassen_block(0, 0, 0, 0, col)
    col = add_strassen_block(0, 0, 0, 2, col)
    col = add_strassen_block(0, 0, 0, 4, col)
    col = add_schoolbook_2x2x1_block(0, 0, 0, 6, col)
    
    # A2 (cols 2-3), B2 (rows 2-3)
    col = add_strassen_block(0, 2, 2, 0, col)
    col = add_strassen_block(0, 2, 2, 2, col)
    col = add_strassen_block(0, 2, 2, 4, col)
    col = add_schoolbook_2x2x1_block(0, 2, 2, 6, col)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
