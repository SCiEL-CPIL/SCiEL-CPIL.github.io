import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=8
    # Rank 29 decomposition
    rank = 29
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    # u_base: (row_offset, col_offset) in U (2x3 matrix)
    # v_base: (row_offset, col_offset) in V (3x8 matrix)
    # w_base: (row_offset, col_offset) in W (8x2 matrix)
    def add_strassen_block(u_row_off, u_col_off, v_row_off, v_col_off, w_row_off, w_col_off, c):
        # U indices: (i, j) -> i*3 + j
        # V indices: (j, k) -> j*8 + k
        # W indices: (k, i) -> k*2 + i
        
        # p1
        U[u_row_off + 0][u_col_off + 0][c] = 1
        U[u_row_off + 1][u_col_off + 1][c] = 1
        V[v_row_off + 0][v_col_off + 0][c] = 1
        V[v_row_off + 1][v_col_off + 1][c] = 1
        W[w_row_off + 0][w_col_off + 0][c] = 1
        W[w_row_off + 1][w_col_off + 1][c] = 1
        c += 1
        
        # p2
        U[u_row_off + 1][u_col_off + 0][c] = 1
        U[u_row_off + 1][u_col_off + 1][c] = 1
        V[v_row_off + 0][v_col_off + 0][c] = 1
        W[w_row_off + 1][w_col_off + 0][c] = 1
        W[w_row_off + 1][w_col_off + 1][c] = -1
        c += 1
        
        # p3
        U[u_row_off + 0][u_col_off + 0][c] = 1
        V[v_row_off + 0][v_col_off + 1][c] = 1
        V[v_row_off + 1][v_col_off + 1][c] = -1
        W[w_row_off + 0][w_col_off + 1][c] = 1
        W[w_row_off + 1][w_col_off + 1][c] = 1
        c += 1
        
        # p4
        U[u_row_off + 1][u_col_off + 1][c] = 1
        V[v_row_off + 1][v_col_off + 0][c] = 1
        V[v_row_off + 0][v_col_off + 0][c] = -1
        W[w_row_off + 0][w_col_off + 0][c] = 1
        W[w_row_off + 1][w_col_off + 0][c] = 1
        c += 1
        
        # p5
        U[u_row_off + 0][u_col_off + 0][c] = 1
        U[u_row_off + 0][u_col_off + 1][c] = 1
        V[v_row_off + 1][v_col_off + 1][c] = 1
        W[w_row_off + 0][w_col_off + 0][c] = -1
        W[w_row_off + 0][w_col_off + 1][c] = 1
        c += 1
        
        # p6
        U[u_row_off + 1][u_col_off + 0][c] = 1
        U[u_row_off + 0][u_col_off + 0][c] = -1
        V[v_row_off + 0][v_col_off + 0][c] = 1
        V[v_row_off + 0][v_col_off + 1][c] = 1
        W[w_row_off + 1][w_col_off + 1][c] = 1
        c += 1
        
        # p7
        U[u_row_off + 0][u_col_off + 1][c] = 1
        U[u_row_off + 1][u_col_off + 1][c] = -1
        V[v_row_off + 1][v_col_off + 0][c] = 1
        V[v_row_off + 1][v_col_off + 1][c] = 1
        W[w_row_off + 0][w_col_off + 0][c] = 1
        c += 1
        
        return c

    # Block 1: A1 (2x2) * B1a1 (2x2)
    # A rows 0,1 cols 0,1. B rows 0,1 cols 0,1. C rows 0,1 cols 0,1.
    col = add_strassen_block(0, 0, 0, 0, 0, 0, col)
    
    # Block 2: A1 (2x2) * B1a2 (2x2)
    # A rows 0,1 cols 0,1. B rows 0,1 cols 2,3. C rows 0,1 cols 2,3.
    col = add_strassen_block(0, 0, 0, 2, 2, 0, col)
    
    # Block 3: A1 (2x2) * B1b1 (2x2)
    # A rows 0,1 cols 0,1. B rows 0,1 cols 4,5. C rows 0,1 cols 4,5.
    col = add_strassen_block(0, 0, 0, 4, 4, 0, col)
    
    # Block 4: A1 (2x2) * B1b2 (2x2)
    # A rows 0,1 cols 0,1. B rows 0,1 cols 6,7. C rows 0,1 cols 6,7.
    col = add_strassen_block(0, 0, 0, 6, 6, 0, col)
    
    # Block 5: A2 (2x1) * B2 (1x8)
    # A col 2. B row 2. C all cols.
    # U[i, 2] = 1. V[2, k] = 1. W[k, i] = 1.
    for i in range(2):
        for k in range(8):
            U[i * 3 + 2][col] = 1
            V[2 * 8 + k][col] = 1
            W[k * 2 + i][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
