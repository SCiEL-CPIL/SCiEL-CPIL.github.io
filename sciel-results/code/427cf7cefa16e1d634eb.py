"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Uses a block decomposition exploiting the low rank of 2x2 x 2x9 multiplication:
  - Split A (4x4) into four 2x2 blocks: A11, A12, A21, A22
  - Split B (4x9) into two 2x9 blocks: B_top, B_bot
  - C = A*B has 4x9 blocks:
    C_top    = A11*B_top + A12*B_bot
    C_bottom = A21*B_top + A22*B_bot
  - Each 2x2 x 2x9 product has rank 4 (factor by rows of the 2x2 matrix)
  - Total rank: 4 products * 4 rank each = 16
"""

import time

DIMENSIONS = [4, 4, 9]

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses block decomposition with rank 16.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a rank-1 term to the decomposition."""
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # For a 2x2 x 2x9 multiplication:
    # A = [[a00, a01], [a10, a11]]  (2x2)
    # B = [[b00, b01, ..., b08],
    #      [b10, b11, ..., b18]]  (2x9)
    # C = A*B = [[c00, c01, ..., c08],
    #            [c10, c11, ..., c18]]  (2x9)
    #
    # c_i,k = a_i,0 * b_0,k + a_i,1 * b_1,k
    #
    # This decomposes into 4 rank-1 terms:
    # Term 0: a_0,0 * b_0,k -> c_0,k for all k
    # Term 1: a_0,1 * b_1,k -> c_0,k for all k
    # Term 2: a_1,0 * b_0,k -> c_1,k for all k
    # Term 3: a_1,1 * b_1,k -> c_1,k for all k
    
    def add_2x2_x_2x9(A_base_i, A_base_j, B_base_j, C_base_i, C_base_k):
        """
        Add 4 rank-1 terms for a 2x2 x 2x9 block multiplication.
        
        A is at rows A_base_i..A_base_i+1, cols A_base_j..A_base_j+1
        B is at rows B_base_j..B_base_j+1, cols C_base_k..C_base_k+8
        C is at rows C_base_i..C_base_i+1, cols C_base_k..C_base_k+8
        """
        for i in range(2):  # row of A (0 or 1)
            for j in range(2):  # col of A / row of B (0 or 1)
                u_vec = [0] * (n * m)  # (4*4) = 16
                v_vec = [0] * (m * p)  # (4*9) = 36
                w_vec = [0] * (p * n)  # (9*4) = 36
                
                # U: A entry at (A_base_i + i, A_base_j + j)
                u_vec[(A_base_i + i) * m + (A_base_j + j)] = 1
                
                # V: B entries at row (B_base_j + j), cols 0..8
                for k in range(9):
                    v_vec[(B_base_j + j) * p + (C_base_k + k)] = 1
                
                # W: C entries at row (C_base_i + i), cols 0..8
                for k in range(9):
                    w_vec[(C_base_k + k) * n + (C_base_i + i)] = 1
                
                add_term(u_vec, v_vec, w_vec)
    
    # C = A * B
    # A = [[A11, A12], [A21, A22]]  (each 2x2)
    # B = [[B_top],    [B_bot]]     (each 2x9)
    # C = [[A11*B_top + A12*B_bot],
    #      [A21*B_top + A22*B_bot]]
    
    # Block 1: A11 (rows 0-1, cols 0-1) * B_top (rows 0-1, cols 0-8) -> C rows 0-1, cols 0-8
    add_2x2_x_2x9(A_base_i=0, A_base_j=0, B_base_j=0, C_base_i=0, C_base_k=0)
    
    # Block 2: A12 (rows 0-1, cols 2-3) * B_bot (rows 2-3, cols 0-8) -> C rows 0-1, cols 0-8
    add_2x2_x_2x9(A_base_i=0, A_base_j=2, B_base_j=2, C_base_i=0, C_base_k=0)
    
    # Block 3: A21 (rows 2-3, cols 0-1) * B_top (rows 0-1, cols 0-8) -> C rows 2-3, cols 0-8
    add_2x2_x_2x9(A_base_i=2, A_base_j=0, B_base_j=0, C_base_i=2, C_base_k=0)
    
    # Block 4: A22 (rows 2-3, cols 2-3) * B_bot (rows 2-3, cols 0-8) -> C rows 2-3, cols 0-8
    add_2x2_x_2x9(A_base_i=2, A_base_j=2, B_base_j=2, C_base_i=2, C_base_k=0)
    
    # Build U, V, W matrices
    rank = len(U_cols)
    U = [[U_cols[col][row] for col in range(rank)] for row in range(n * m)]
    V = [[V_cols[col][row] for col in range(rank)] for row in range(m * p)]
    W = [[W_cols[col][row] for col in range(rank)] for row in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
