import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    
    # Strassen 2x2x2 decomposition
    # U, V, W are 2x2 matrices
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # U vectors (linear forms on A)
    U_s = [
        [[1, 0], [0, 1]], # A11+A22
        [[0, 0], [1, 1]], # A21+A22
        [[1, 0], [0, 0]], # A11
        [[0, 0], [0, 1]], # A22
        [[1, 1], [0, 0]], # A11+A12
        [[-1, 0], [1, 0]],# A21-A11
        [[0, 1], [0, -1]] # A12-A22
    ]
    
    # V vectors (linear forms on B)
    V_s = [
        [[1, 0], [0, 1]], # B11+B22
        [[1, 0], [0, 0]], # B11
        [[0, 1], [0, -1]],# B12-B22
        [[-1, 0], [1, 0]],# B21-B11
        [[0, 0], [0, 1]], # B22
        [[1, 1], [0, 0]], # B11+B12
        [[0, 0], [1, 1]]  # B21+B22
    ]
    
    # W vectors (linear forms on C)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    W_s = [
        [[1, 0], [0, 1]], # M1: C11+1, C22+1
        [[0, 0], [1, -1]],# M2: C21+1, C22-1
        [[0, 1], [0, 1]], # M3: C12+1, C22+1
        [[1, 0], [1, 0]], # M4: C11+1, C21+1
        [[-1, 1], [0, 0]],# M5: C11-1, C12+1
        [[0, 0], [0, 1]], # M6: C22+1
        [[1, 0], [0, 0]]  # M7: C11+1
    ]

    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 2 blocks of A (A1, A2)
    # 2 blocks of B rows (B1, B2)
    # 3 blocks of B cols (B_1, B_2, B_3)
    
    for a_block in range(2): # 0 for A1, 1 for A2
        for b_row_block in range(2): # 0 for B1, 1 for B2
            for b_col_block in range(3): # 0, 1, 2
                for s in range(7):
                    # Embed U_s[s] into 2x4
                    # A_block 0: cols 0,1. A_block 1: cols 2,3.
                    u_mat = [[0]*4 for _ in range(2)]
                    u_s_mat = U_s[s]
                    for i in range(2):
                        for j in range(2):
                            u_mat[i][a_block*2 + j] = u_s_mat[i][j]
                            
                    # Embed V_s[s] into 4x6
                    # B_row_block 0: rows 0,1. B_row_block 1: rows 2,3.
                    # B_col_block j: cols 2*j, 2*j+1.
                    v_mat = [[0]*6 for _ in range(4)]
                    v_s_mat = V_s[s]
                    for i in range(2):
                        for j in range(2):
                            v_mat[b_row_block*2 + i][b_col_block*2 + j] = v_s_mat[i][j]
                            
                    # Embed W_s[s] into 2x6 (for C)
                    # C is 2x6.
                    # W_s[s] is 2x2.
                    # It corresponds to cols b_col_block*2, b_col_block*2+1 of C.
                    w_mat = [[0]*6 for _ in range(2)]
                    w_s_mat = W_s[s]
                    for i in range(2):
                        for j in range(2):
                            w_mat[i][b_col_block*2 + j] = w_s_mat[i][j]
                            
                    # Fill U, V, W arrays
                    # U index: i*m + j. Here m=4.
                    for i in range(2):
                        for j in range(4):
                            U[i*4 + j][col] = u_mat[i][j]
                            
                    # V index: i*p + j? No, V is (m*p, rank).
                    # m=4, p=6. Index is row*m? No.
                    # V corresponds to B. B is 4x6.
                    # Index in V is (row_B * p + col_B)?
                    # Task says V shape (m*p, rank). m=4, p=6.
                    # Usually V[j*p+k] where j is row, k is col?
                    # Let's check seed: V[j*p+k][column] = 1.
                    # j in range(m), k in range(p).
                    # So V index is j*p + k.
                    for i in range(4):
                        for j in range(6):
                            V[i*6 + j][col] = v_mat[i][j]
                            
                    # W index: k*n + i.
                    # Task says W[k*n+i, r].
                    # k is col of B (0..p-1), i is row of A (0..n-1).
                    # W_mat is C. C is n x p.
                    # C[i][k] = sum ...
                    # So W[k*n + i] corresponds to C[i][k].
                    for i in range(2): # row of C
                        for k in range(6): # col of C
                            W[k*2 + i][col] = w_mat[i][k]
                            
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
