import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # We use the block decomposition described:
    # Rank 68 total.
    
    # Helper to add a rank-1 term
    U = [[0] * 68 for _ in range(n * m)]
    V = [[0] * 68 for _ in range(m * p)]
    W = [[0] * 68 for _ in range(p * n)]
    
    col = 0
    
    # Function to add Strassen 2x2x2 decomposition
    # Local indices: A (2x2), B (2x2), C (2x2)
    # u_idx[i, j], v_idx[i, j], w_idx[i, j] are global indices
    def add_strassen(u_base, v_base, w_base):
        nonlocal col
        # M1
        U[u_base[0][0]][col] += 1
        U[u_base[1][1]][col] += 1
        V[v_base[0][0]][col] += 1
        V[v_base[1][1]][col] += 1
        W[w_base[0][0]][col] += 1
        W[w_base[1][1]][col] += 1
        col += 1
        
        # M2
        U[u_base[1][0]][col] += 1
        U[u_base[1][1]][col] += 1
        V[v_base[0][0]][col] += 1
        W[w_base[1][0]][col] += 1
        col += 1
        
        # M3
        U[u_base[0][0]][col] += 1
        V[v_base[0][1]][col] += 1
        V[v_base[1][1]][col] += -1
        W[w_base[0][1]][col] += 1
        col += 1
        
        # M4
        U[u_base[1][1]][col] += 1
        V[v_base[1][0]][col] += 1
        V[v_base[0][0]][col] += -1
        W[w_base[0][0]][col] += 1
        W[w_base[1][0]][col] += 1
        col += 1
        
        # M5
        U[u_base[0][0]][col] += 1
        U[u_base[0][1]][col] += 1
        V[v_base[1][1]][col] += 1
        W[w_base[0][0]][col] += -1
        W[w_base[0][1]][col] += 1
        col += 1
        
        # M6
        U[u_base[1][0]][col] += 1
        U[u_base[0][0]][col] += -1
        V[v_base[0][0]][col] += 1
        V[v_base[0][1]][col] += 1
        W[w_base[1][1]][col] += 1
        col += 1
        
        # M7
        U[u_base[0][1]][col] += 1
        U[u_base[1][1]][col] += -1
        V[v_base[1][0]][col] += 1
        V[v_base[1][1]][col] += 1
        W[w_base[0][0]][col] += 1
        W[w_base[1][1]][col] += -1
        col += 1

    # Function to add Schoolbook 2x2x1 decomposition (for 3rd column of 2x2x3)
    def add_schoolbook_2x2x1(u_base, v_base, w_base):
        nonlocal col
        # A is 2x2, B is 2x1 (col 2), C is 2x1
        # C_0 = A_00 B_02 + A_01 B_12
        # C_1 = A_10 B_02 + A_11 B_12
        
        U[u_base[0][0]][col] += 1
        V[v_base[0][2]][col] += 1
        W[w_base[2][0]][col] += 1
        col += 1
        
        U[u_base[0][1]][col] += 1
        V[v_base[1][2]][col] += 1
        W[w_base[2][0]][col] += 1
        col += 1
        
        U[u_base[1][0]][col] += 1
        V[v_base[0][2]][col] += 1
        W[w_base[2][1]][col] += 1
        col += 1
        
        U[u_base[1][1]][col] += 1
        V[v_base[1][2]][col] += 1
        W[w_base[2][1]][col] += 1
        col += 1

    # Function to add Schoolbook 1x2x3 decomposition
    def add_schoolbook_1x2x3(u_base, v_base, w_base):
        nonlocal col
        # A is 1x2, B is 2x3, C is 1x3
        # C_0 = A_00 B_00 + A_01 B_10
        # C_1 = A_00 B_01 + A_01 B_11
        # C_2 = A_00 B_02 + A_01 B_12
        
        for k in range(3):
            U[u_base[0][0]][col] += 1
            V[v_base[0][k]][col] += 1
            W[w_base[k][0]][col] += 1
            col += 1
            
            U[u_base[0][1]][col] += 1
            V[v_base[1][k]][col] += 1
            W[w_base[k][0]][col] += 1
            col += 1

    # Block 1: A11 B11 (2x2 A, 2x3 B)
    # A11: i 0,1; j 0,1
    # B11: j 0,1; k 0,1,2
    # C11: i 0,1; k 0,1,2
    u_b1 = [[i*4 + j for j in range(2)] for i in range(2)]
    v_b1 = [[j*6 + k for k in range(3)] for j in range(2)]
    w_b1 = [[k*3 + i for i in range(2)] for k in range(3)]
    add_strassen(u_b1, v_b1, w_b1)
    add_schoolbook_2x2x1(u_b1, v_b1, w_b1)

    # Block 2: A11 B12 (2x2 A, 2x3 B)
    # A11: i 0,1; j 0,1
    # B12: j 0,1; k 3,4,5
    # C12: i 0,1; k 3,4,5
    u_b2 = [[i*4 + j for j in range(2)] for i in range(2)]
    v_b2 = [[j*6 + (k+3) for k in range(3)] for j in range(2)]
    w_b2 = [[(k+3)*3 + i for i in range(2)] for k in range(3)]
    add_strassen(u_b2, v_b2, w_b2)
    add_schoolbook_2x2x1(u_b2, v_b2, w_b2)

    # Block 3: A12 B21 (2x2 A, 2x3 B)
    # A12: i 0,1; j 2,3
    # B21: j 2,3; k 0,1,2
    # C11: i 0,1; k 0,1,2
    u_b3 = [[i*4 + (j+2) for j in range(2)] for i in range(2)]
    v_b3 = [[(j+2)*6 + k for k in range(3)] for j in range(2)]
    w_b3 = [[k*3 + i for i in range(2)] for k in range(3)]
    add_strassen(u_b3, v_b3, w_b3)
    add_schoolbook_2x2x1(u_b3, v_b3, w_b3)

    # Block 4: A12 B22 (2x2 A, 2x3 B)
    # A12: i 0,1; j 2,3
    # B22: j 2,3; k 3,4,5
    # C12: i 0,1; k 3,4,5
    u_b4 = [[i*4 + (j+2) for j in range(2)] for i in range(2)]
    v_b4 = [[(j+2)*6 + (k+3) for k in range(3)] for j in range(2)]
    w_b4 = [[(k+3)*3 + i for i in range(2)] for k in range(3)]
    add_strassen(u_b4, v_b4, w_b4)
    add_schoolbook_2x2x1(u_b4, v_b4, w_b4)

    # Block 5: A21 B11 (1x2 A, 2x3 B)
    # A21: i 2; j 0,1
    # B11: j 0,1; k 0,1,2
    # C21: i 2; k 0,1,2
    u_b5 = [[2*4 + j for j in range(2)]]
    v_b5 = [[j*6 + k for k in range(3)] for j in range(2)]
    w_b5 = [[k*3 + 2 for i in range(1)] for k in range(3)]
    add_schoolbook_1x2x3(u_b5, v_b5, w_b5)

    # Block 6: A21 B12 (1x2 A, 2x3 B)
    # A21: i 2; j 0,1
    # B12: j 0,1; k 3,4,5
    # C22: i 2; k 3,4,5
    u_b6 = [[2*4 + j for j in range(2)]]
    v_b6 = [[j*6 + (k+3) for k in range(3)] for j in range(2)]
    w_b6 = [[(k+3)*3 + 2 for i in range(1)] for k in range(3)]
    add_schoolbook_1x2x3(u_b6, v_b6, w_b6)

    # Block 7: A22 B21 (1x2 A, 2x3 B)
    # A22: i 2; j 2,3
    # B21: j 2,3; k 0,1,2
    # C21: i 2; k 0,1,2
    u_b7 = [[2*4 + (j+2) for j in range(2)]]
    v_b7 = [[(j+2)*6 + k for k in range(3)] for j in range(2)]
    w_b7 = [[k*3 + 2 for i in range(1)] for k in range(3)]
    add_schoolbook_1x2x3(u_b7, v_b7, w_b7)

    # Block 8: A22 B22 (1x2 A, 2x3 B)
    # A22: i 2; j 2,3
    # B22: j 2,3; k 3,4,5
    # C22: i 2; k 3,4,5
    u_b8 = [[2*4 + (j+2) for j in range(2)]]
    v_b8 = [[(j+2)*6 + (k+3) for k in range(3)] for j in range(2)]
    w_b8 = [[(k+3)*3 + 2 for i in range(1)] for k in range(3)]
    add_schoolbook_1x2x3(u_b8, v_b8, w_b8)

    return {"U": U, "V": V, "W": W}
