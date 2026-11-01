import time
import sys

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-28 exact decomposition for 2x3x5 matrix multiplication.
    """
    n, m, p = 2, 3, 5
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)] # 6 x 28
    V = [[0] * rank for _ in range(m * p)] # 15 x 28
    W = [[0] * rank for _ in range(p * n)] # 10 x 28
    
    # Part 1: [2, 1, 5] decomposition (Rank 10)
    # Handles A[:, 2] and B[2, :]
    # A indices: 3*i + 2
    # B indices: 10 + k
    # C indices: 2*k + i
    col = 0
    for i in range(2):
        for k in range(5):
            U[3*i + 2][col] = 1
            V[10 + k][col] = 1
            W[2*k + i][col] = 1
            col += 1
            
    # Part 2: [2, 2, 5] decomposition (Rank 18)
    # Handles A[:, 0:2] and B[0:2, :]
    # A indices: 3*i + j for j in 0,1
    # B indices: 5*j + k for j in 0,1
    # C indices: 2*k + i
    
    # Subpart 2a: k in {0, 1} -> [2, 2, 2] Laderman (Rank 7)
    # A: 0,1,3,4
    # B: 0,1,5,6
    # C: 0,1,2,3
    
    def add_laderman(a_idx, b_idx, c_idx, start_col):
        # a_idx: [a00, a01, a10, a11]
        # b_idx: [b00, b01, b10, b11]
        # c_idx: [c00, c01, c10, c11]
        
        # M1 = a00 * (b10 - b00)
        U[a_idx[0]][start_col] = 1
        V[b_idx[2]][start_col] = 1
        V[b_idx[0]][start_col] = -1
        W[c_idx[1]][start_col] = 1
        W[c_idx[3]][start_col] = 1
        W[c_idx[2]][start_col] = 1
        col1 = start_col + 1
        
        # M2 = (a00 + a01) * b00
        U[a_idx[0]][col1] = 1
        U[a_idx[1]][col1] = 1
        V[b_idx[0]][col1] = 1
        W[c_idx[0]][col1] = 1
        W[c_idx[1]][col1] = 1
        col2 = start_col + 2
        
        # M3 = (a10 + a11) * b11
        U[a_idx[2]][col2] = 1
        U[a_idx[3]][col2] = 1
        V[b_idx[3]][col2] = 1
        W[c_idx[2]][col2] = 1
        W[c_idx[3]][col2] = 1
        col3 = start_col + 3
        
        # M4 = a11 * (b01 - b11)
        U[a_idx[3]][col3] = 1
        V[b_idx[1]][col3] = 1
        V[b_idx[3]][col3] = -1
        W[c_idx[0]][col3] = 1
        W[c_idx[3]][col3] = 1
        col4 = start_col + 4
        
        # M5 = (a00 + a10) * b10
        U[a_idx[0]][col4] = 1
        U[a_idx[2]][col4] = 1
        V[b_idx[2]][col4] = 1
        W[c_idx[0]][col4] = -1
        W[c_idx[2]][col4] = 1
        col5 = start_col + 5
        
        # M6 = (a01 - a10) * (b00 + b10)
        U[a_idx[1]][col5] = 1
        U[a_idx[2]][col5] = -1
        V[b_idx[0]][col5] = 1
        V[b_idx[2]][col5] = 1
        W[c_idx[1]][col5] = -1
        W[c_idx[3]][col5] = 1
        col6 = start_col + 6
        
        # M7 = (a01 + a11) * (b11 - b01)
        U[a_idx[1]][col6] = 1
        U[a_idx[3]][col6] = 1
        V[b_idx[3]][col6] = 1
        V[b_idx[1]][col6] = -1
        W[c_idx[0]][col6] = 1
        W[c_idx[1]][col6] = 1
        W[c_idx[2]][col6] = -1
        W[c_idx[3]][col6] = 1
        
        return start_col + 7

    # Indices for Subpart 2a
    a_idx_2a = [0, 1, 3, 4] # a00, a01, a10, a11
    b_idx_2a = [0, 1, 5, 6] # b00, b01, b10, b11
    c_idx_2a = [0, 1, 2, 3] # c00, c01, c10, c11
    
    col = add_laderman(a_idx_2a, b_idx_2a, c_idx_2a, col)
    
    # Subpart 2b: k in {2, 3} -> [2, 2, 2] Laderman (Rank 7)
    # A same
    # B: 2,3,7,8
    # C: 4,5,6,7
    
    a_idx_2b = [0, 1, 3, 4]
    b_idx_2b = [2, 3, 7, 8] # b02, b03, b12, b13
    c_idx_2b = [4, 5, 6, 7] # c02, c03, c12, c13
    
    col = add_laderman(a_idx_2b, b_idx_2b, c_idx_2b, col)
    
    # Subpart 2c: k in {4} -> [2, 2, 1] Schoolbook (Rank 4)
    # A same
    # B: 4, 9 (b04, b14)
    # C: 8, 9 (c04, c14)
    
    # Term 1: A00 * B04 -> C04
    U[0][col] = 1
    V[4][col] = 1
    W[8][col] = 1
    col += 1
    
    # Term 2: A01 * B14 -> C04
    U[1][col] = 1
    V[9][col] = 1
    W[8][col] = 1
    col += 1
    
    # Term 3: A10 * B04 -> C14
    U[3][col] = 1
    V[4][col] = 1
    W[9][col] = 1
    col += 1
    
    # Term 4: A11 * B14 -> C14
    U[4][col] = 1
    V[9][col] = 1
    W[9][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
