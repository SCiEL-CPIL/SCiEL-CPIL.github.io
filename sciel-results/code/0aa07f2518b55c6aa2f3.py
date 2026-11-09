import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 matrix multiplication with Strassen's rank-7 algorithm.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen's 7 products for C = A*B where A,B are 2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    #
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    #
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # Column 0: m1 = (a11+a22)(b11+b22)
    # U: a11 at (0,0), a22 at (1,1)
    U[0][0] = 1; U[3][0] = 1
    # V: b11 at (0,0), b22 at (1,1)
    V[0][0] = 1; V[3][0] = 1
    # W: c11 += m1, c22 += m1
    W[0][0] = 1; W[3][0] = 1  # W[k*2+i]: c11 is k=0,i=0 -> idx 0; c22 is k=1,i=1 -> idx 3
    
    # Column 1: m2 = (a21+a22)b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # c21: k=1,i=0 -> idx 2
    W[3][1] = -1  # c22: k=1,i=1 -> idx 3
    
    # Column 2: m3 = a11(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # c12: k=0,i=1 -> idx 1
    W[3][2] = 1  # c22: k=1,i=1 -> idx 3
    
    # Column 3: m4 = a22(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1  # c11: k=0,i=0 -> idx 0
    W[2][3] = 1  # c21: k=1,i=0 -> idx 2
    
    # Column 4: m5 = (a11+a12)b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1  # c11: k=0,i=0 -> idx 0
    W[1][4] = 1  # c12: k=0,i=1 -> idx 1
    
    # Column 5: m6 = (a21-a11)(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # c22: k=1,i=1 -> idx 3
    
    # Column 6: m7 = (a12-a22)(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # c11: k=0,i=0 -> idx 0
    
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """Return (U, V, W) for n x m times m x p schoolbook multiplication.
    U shape: (n*m, n*m*p), V shape: (m*p, n*m*p), W shape: (p*n, n*m*p)
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _combine_blocks(block_decomps, n_total, m_total, p_total, 
                     a_row_blocks, a_col_blocks, b_row_blocks, b_col_blocks,
                     c_row_blocks, c_col_blocks):
    """
    Combine block decompositions into a full decomposition.
    
    Each block_decomp is (U_b, V_b, W_b) for a sub-multiplication.
    a_row_blocks: list of (start, end) row ranges in A for each block
    a_col_blocks: list of (start, end) col ranges in A for each block  
    b_row_blocks: list of (start, end) row ranges in B for each block
    b_col_blocks: list of (start, end) col ranges in B for each block
    c_row_blocks: list of (start, end) row ranges in C for each block
    c_col_blocks: list of (start, end) col ranges in C for each block
    """
    total_rank = sum(len(U_b[0]) for U_b, V_b, W_b in block_decomps)
    
    U = [[0]*total_rank for _ in range(n_total * m_total)]
    V = [[0]*total_rank for _ in range(m_total * p_total)]
    W = [[0]*total_rank for _ in range(p_total * n_total)]
    
    col_offset = 0
    for idx, (U_b, V_b, W_b) in enumerate(block_decomps):
        rank_b = len(U_b[0])
        a_r0, a_r1 = a_row_blocks[idx]
        a_c0, a_c1 = a_col_blocks[idx]
        b_r0, b_r1 = b_row_blocks[idx]
        b_c0, b_c1 = b_col_blocks[idx]
        c_r0, c_r1 = c_row_blocks[idx]
        c_c0, c_c1 = c_col_blocks[idx]
        
        for i in range(len(U_b)):
            local_i = i
            global_i = (a_r0 + local_i // m_total) * m_total + (a_c0 + local_i % m_total)
            for c in range(rank_b):
                U[global_i][col_offset + c] = U_b[local_i][c]
        
        for j in range(len(V_b)):
            local_j = j
            global_j = (b_r0 + local_j // p_total) * p_total + (b_c0 + local_j % p_total)
            for c in range(rank_b):
                V[global_j][col_offset + c] = V_b[local_j][c]
        
        for k in range(len(W_b)):
            local_k = k
            global_k = (c_c0 + local_k // n_total) * n_total + (c_r0 + local_k % n_total)
            for c in range(rank_b):
                W[global_k][col_offset + c] = W_b[local_k][c]
        
        col_offset += rank_b
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Strategy: Split A (3x4) into top 2x4 and bottom 1x4
    # Split B (4x5) into left 4x2 and right 4x3
    # C = A*B gives:
    #   C_top (2x5) = A_top (2x4) * B (4x5)
    #   C_bot (1x5) = A_bot (1x4) * B (4x5)
    #
    # Further split B into B_left (4x2) and B_right (4x3):
    #   C_top_left (2x2) = A_top (2x4) * B_left (4x2)
    #   C_top_right (2x3) = A_top (2x4) * B_right (4x3)
    #   C_bot_left (1x2) = A_bot (1x4) * B_left (4x2)
    #   C_bot_right (1x3) = A_bot (1x4) * B_right (4x3)
    #
    # For 2x4x2: split A_top into 2x2 + 2x2, B_left into 2x2 + 2x2
    #   (2x2)*(2x2) + (2x2)*(2x2) -> 7 + 7 = 14
    # For 2x4x3: split A_top into 2x2 + 2x2, B_right into 2x3 + 2x3
    #   (2x2)*(2x3) + (2x2)*(2x3)
    #   2x2x3 rank: use Strassen on first 2 cols (rank 7) + schoolbook for 3rd col (rank 4) = 11
    #   So 11 + 11 = 22
    # For 1x4x2: split A_bot into 1x2 + 1x2, B_left into 2x2 + 2x2
    #   (1x2)*(2x2) + (1x2)*(2x2) -> 4 + 4 = 8
    # For 1x4x3: split A_bot into 1x2 + 1x2, B_right into 2x3 + 2x3
    #   (1x2)*(2x3) + (1x2)*(2x3) -> 6 + 6 = 12
    #
    # Total: 14 + 22 + 8 + 12 = 56
    
    # Helper: 2x2x3 decomposition (Strassen on first 2 cols + schoolbook on 3rd)
    def _strassen_2x2x3():
        """2x2 times 2x3. Rank 11."""
        U222, V222, W222 = _strassen_2x2x2()
        # Add 3rd column of B (b13, b23)
        # c13 = a11*b13 + a12*b23
        # c23 = a21*b13 + a22*b23
        U = [row[:] for row in U222]
        V = [row[:] for row in V222]
        W = [row[:] for row in W222]
        
        for row in U:
            row.extend([0, 0])
        for row in V:
            row.extend([0, 0])
        for row in W:
            row.extend([0, 0])
        
        # Column 7: a11*b13
        U[0][7] = 1
        V[0][7] = 1
        W[1][7] = 1  # c13: k=0,i=1 -> idx 1
        
        # Column 8: a12*b23
        U[1][8] = 1
        V[1][8] = 1
        W[1][8] = 1  # c13: k=0,i=1 -> idx 1
        
        # Column 9: a21*b13
        U[2][9] = 1
        V[0][9] = 1
        W[4][9] = 1  # c23: k=1,i=0 -> idx 4
        
        # Column 10: a22*b23
        U[3][10] = 1
        V[1][10] = 1
        W[4][10] = 1  # c23: k=1,i=0 -> idx 4
        
        return U, V, W
    
    # Helper: 1x2x2 (rank 4)
    def _schoolbook_1x2x2():
        U = [[0]*4 for _ in range(2)]
        V = [[0]*4 for _ in range(4)]
        W = [[0]*4 for _ in range(2)]
        for j in range(2):
            for k in range(2):
                col = j*2+k
                U[j][col] = 1
                V[j*2+k][col] = 1
                W[k][col] = 1
        return U, V, W
    
    # Helper: 1x2x3 (rank 6)
    def _schoolbook_1x2x3():
        U = [[0]*6 for _ in range(2)]
        V = [[0]*6 for _ in range(6)]
        W = [[0]*6 for _ in range(3)]
        for j in range(2):
            for k in range(3):
                col = j*3+k
                U[j][col] = 1
                V[j*3+k][col] = 1
                W[k][col] = 1
        return U, V, W
    
    # Block 1: A_top_left (2x2) * B_top_left (2x2) -> C_top_left_top (2x2)
    # A rows 0-1, cols 0-1; B rows 0-1, cols 0-1; C rows 0-1, cols 0-1
    U1, V1, W1 = _strassen_2x2x2()
    
    # Block 2: A_top_right (2x2) * B_bot_left (2x2) -> C_top_left_bot (2x2)
    # A rows 0-1, cols 2-3; B rows 2-3, cols 0-1; C rows 0-1, cols 0-1
    U2, V2, W2 = _strassen_2x2x2()
    
    # Block 3: A_top_left (2x2) * B_top_right (2x3) -> C_top_right_top (2x3)
    # A rows 0-1, cols 0-1; B rows 0-1, cols 2-4; C rows 0-1, cols 2-4
    U3, V3, W3 = _strassen_2x2x3()
    
    # Block 4: A_top_right (2x2) * B_bot_right (2x3) -> C_top_right_bot (2x3)
    # A rows 0-1, cols 2-3; B rows 2-3, cols 2-4; C rows 0-1, cols 2-4
    U4, V4, W4 = _strassen_2x2x3()
    
    # Block 5: A_bot_left (1x2) * B_top_left (2x2) -> C_bot_left_top (1x2)
    # A row 2, cols 0-1; B rows 0-1, cols 0-1; C row 2, cols 0-1
    U5, V5, W5 = _schoolbook_1x2x2()
    
    # Block 6: A_bot_right (1x2) * B_bot_left (2x2) -> C_bot_left_bot (1x2)
    # A row 2, cols 2-3; B rows 2-3, cols 0-1; C row 2, cols 0-1
    U6, V6, W6 = _schoolbook_1x2x2()
    
    # Block 7: A_bot_left (1x2) * B_top_right (2x3) -> C_bot_right_top (1x3)
    # A row 2, cols 0-1; B rows 0-1, cols 2-4; C row 2, cols 2-4
    U7, V7, W7 = _schoolbook_1x2x3()
    
    # Block 8: A_bot_right (1x2) * B_bot_right (2x3) -> C_bot_right_bot (1x3)
    # A row 2, cols 2-3; B rows 2-3, cols 2-4; C row 2, cols 2-4
    U8, V8, W8 = _schoolbook_1x2x3()
    
    # Now assemble. n=3, m=4, p=5
    # A is 3x4: rows 0,1,2; cols 0,1,2,3
    # B is 4x5: rows 0,1,2,3; cols 0,1,2,3,4
    # C is 3x5: rows 0,1,2; cols 0,1,2,3,4
    
    # U: (n*m, rank) = (12, total_rank)
    # V: (m*p, rank) = (20, total_rank)
    # W: (p*n, rank) = (15, total_rank)
    
    total_rank = 7 + 7 + 11 + 11 + 4 + 4 + 6 + 6  # = 56
    U = [[0]*total_rank for _ in range(12)]
    V = [[0]*total_rank for _ in range(20)]
    W = [[0]*total_rank for _ in range(15)]
    
    col = 0
    # Block 1: A rows 0-1, cols 0-1 -> U indices 0,1,4,5 (i*m+j: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5)
    #           B rows 0-1, cols 0-1 -> V indices 0,1,2,3 (j*p+k: 0*5+0=0, 0*5+1=1, 1*5+0=5, 1*5+1=6)
    #           C rows 0-1, cols 0-1 -> W indices k*n+i: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4
    for i in range(4):
        for c in range(7):
            U[i][col+c] = U1[i][c]
    for j in range(4):
        for c in range(7):
            V[j][col+c] = V1[j][c]
    for k in range(4):
        for c in range(7):
            W[k][col+c] = W1[k][c]
    col += 7
    
    # Block 2: A rows 0-1, cols 2-3 -> U indices 2,3,6,7
    #           B rows 2-3, cols 0-1 -> V indices 10,11,15,16
    #           C rows 0-1, cols 0-1 -> W indices 0,1,3,4
    for i in range(4):
        for c in range(7):
            U[2+i][col+c] = U2[i][c]
    for j in range(4):
        for c in range(7):
            V[10+j][col+c] = V2[j][c]
    for k in range(4):
        for c in range(7):
            W[k][col+c] = W2[k][c]
    col += 7
    
    # Block 3: A rows 0-1, cols 0-1 -> U indices 0,1,4,5
    #           B rows 0-1, cols 2-4 -> V indices 2,3,4,7,8,9
    #           C rows 0-1, cols 2-4 -> W indices: k*n+i for k=0,1; i=0,1; but cols 2-4 means c_col offset
    #           W index = k*n + i, but we need to map local col (0,1,2) to global col (2,3,4)
    #           W[k*n+i] where k is local col index... wait, W index is k*n+i where k is the col index in C
    #           For C rows 0-1, cols 2-4: W index = (2+local_k)*n + local_i = (2+local_k)*3 + local_i
    for i in range(4):
        for c in range(11):
            U[i][col+c] = U3[i][c]
    for j in range(6):
        for c in range(11):
            V[2+j][col+c] = V3[j][c]
    for k in range(6):
        local_k = k  # 0,1,2,3,4,5
        # W3 has 6 rows: k_local*n + i_local where k_local in 0..2, i_local in 0..1
        # Actually W3 shape is (p*n, rank) = (3*2, 11) = (6, 11)
        # W3[k_local*2 + i_local, c]
        # Global W index: (2 + k_local)*3 + i_local
        for c in range(11):
            k_local = k // 2
            i_local = k % 2
            global_w = (2 + k_local) * 3 + i_local
            W[global_w][col+c] = W3[k][c]
    col += 11
    
    # Block 4: A rows 0-1, cols 2-3 -> U indices 2,3,6,7
    #           B rows 2-3, cols 2-4 -> V indices 12,13,14,17,18,19
    #           C rows 0-1, cols 2-4 -> W indices: (2+k_local)*3 + i_local
    for i in range(4):
        for c in range(11):
            U[2+i][col+c] = U4[i][c]
    for j in range(6):
        for c in range(11):
            V[12+j][col+c] = V4[j][c]
    for k in range(6):
        k_local = k // 2
        i_local = k % 2
        global_w = (2 + k_local) * 3 + i_local
        for c in range(11):
            W[global_w][col+c] = W4[k][c]
    col += 11
    
    # Block 5: A row 2, cols 0-1 -> U indices 8,9
    #           B rows 0-1, cols 0-1 -> V indices 0,1,5,6
    #           C row 2, cols 0-1 -> W indices: k*3+2 for k=0,1 -> 2, 5
    for i in range(2):
        for c in range(4):
            U[8+i][col+c] = U5[i][c]
    for j in range(4):
        for c in range(4):
            V[j][col+c] = V5[j][c]
    for k in range(2):
        for c in range(4):
            W[k*3+2][col+c] = W5[k][c]
    col += 4
    
    # Block 6: A row 2, cols 2-3 -> U indices 10,11
    #           B rows 2-3, cols 0-1 -> V indices 10,11,15,16
    #           C row 2, cols 0-1 -> W indices: 2, 5
    for i in range(2):
        for c in range(4):
            U[10+i][col+c] = U6[i][c]
    for j in range(4):
        for c in range(4):
            V[10+j][col+c] = V6[j][c]
    for k in range(2):
        for c in range(4):
            W[k*3+2][col+c] = W6[k][c]
    col += 4
    
    # Block 7: A row 2, cols 0-1 -> U indices 8,9
    #           B rows 0-1, cols 2-4 -> V indices 2,3,4,7,8,9
    #           C row 2, cols 2-4 -> W indices: (2+k_local)*3+2 for k_local=0,1,2 -> 8, 11, 14
    for i in range(2):
        for c in range(6):
            U[8+i][col+c] = U7[i][c]
    for j in range(6):
        for c in range(6):
            V[2+j][col+c] = V7[j][c]
    for k in range(3):
        for c in range(6):
            W[(2+k)*3+2][col+c] = W7[k][c]
    col += 6
    
    # Block 8: A row 2, cols 2-3 -> U indices 10,11
    #           B rows 2-3, cols 2-4 -> V indices 12,13,14,17,18,19
    #           C row 2, cols 2-4 -> W indices: 8, 11, 14
    for i in range(2):
        for c in range(6):
            U[10+i][col+c] = U8[i][c]
    for j in range(6):
        for c in range(6):
            V[12+j][col+c] = V8[j][c]
    for k in range(3):
        for c in range(6):
            W[(2+k)*3+2][col+c] = W8[k][c]
    col += 6
    
    return {"U": U, "V": V, "W": W}
