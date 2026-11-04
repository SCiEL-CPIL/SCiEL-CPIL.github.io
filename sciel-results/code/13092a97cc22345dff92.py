import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        # a_r, a_c, b_r, b_c, c_r, c_c are lists of 2 indices
        # A indices: (a_r[0], a_c[0]) -> U[a_r[0]*m + a_c[0]]
        # B indices: (b_r[0], b_c[0]) -> V[b_r[0]*p + b_c[0]]
        # C indices: (c_r[0], c_c[0]) -> W[c_c[0]*n + c_r[0]]
        
        A00 = a_r[0] * m + a_c[0]
        A01 = a_r[0] * m + a_c[1]
        A10 = a_r[1] * m + a_c[0]
        A11 = a_r[1] * m + a_c[1]
        
        B00 = b_r[0] * p + b_c[0]
        B01 = b_r[0] * p + b_c[1]
        B10 = b_r[1] * p + b_c[0]
        B11 = b_r[1] * p + b_c[1]
        
        C00 = c_c[0] * n + c_r[0]
        C01 = c_c[1] * n + c_r[0]
        C10 = c_c[0] * n + c_r[1]
        C11 = c_c[1] * n + c_r[1]
        
        # M1
        U[A00][col] += 1
        U[A11][col] += 1
        V[B00][col] += 1
        V[B11][col] += 1
        W[C00][col] += 1
        W[C11][col] += 1
        col += 1
        
        # M2
        U[A10][col] += 1
        U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1
        W[C11][col] += -1
        col += 1
        
        # M3
        U[A00][col] += 1
        V[B01][col] += 1
        V[B11][col] += -1
        W[C01][col] += 1
        W[C11][col] += 1
        col += 1
        
        # M4
        U[A11][col] += 1
        V[B10][col] += 1
        V[B00][col] += -1
        W[C00][col] += 1
        W[C10][col] += 1
        col += 1
        
        # M5
        U[A00][col] += 1
        U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] += -1
        W[C01][col] += 1
        col += 1
        
        # M6
        U[A10][col] += 1
        U[A00][col] += -1
        V[B00][col] += 1
        V[B01][col] += 1
        W[C11][col] += 1
        W[C10][col] += -1
        col += 1
        
        # M7
        U[A01][col] += 1
        U[A11][col] += -1
        V[B10][col] += 1
        V[B11][col] += 1
        W[C00][col] += 1
        W[C11][col] += -1
        col += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])
    
    # Block 5: A[0:2, 0:4] * B[0:4, 4:5] -> C[0:2, 4:5]
    # This is M_{2,4,1}
    for i in range(2):
        for j in range(4):
            U[i * m + j][col] += 1
            V[j * p + 4][col] += 1
            W[4 * n + i][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
