import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Rank 33 decomposition
    # Part 1: k=0 (12 terms)
    # Part 2: k=1,2 (21 terms using Strassen on 2x2 blocks)
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)] # 6 x 33
    V = [[0] * rank for _ in range(m * p)] # 18 x 33
    W = [[0] * rank for _ in range(p * n)] # 12 x 33
    
    col = 0
    
    # Part 1: A[:, 0] * B[0, :]
    # A[0,0] -> U[0], A[1,0] -> U[3]
    # B[0,k] -> V[k]
    # C[0,k] -> W[2k], C[1,k] -> W[2k+1]
    for k in range(p):
        # Term 1: A[0,0] * B[0,k] -> C[0,k]
        U[0][col] = 1
        V[k][col] = 1
        W[2*k][col] = 1
        col += 1
        
        # Term 2: A[1,0] * B[0,k] -> C[1,k]
        U[3][col] = 1
        V[k][col] = 1
        W[2*k + 1][col] = 1
        col += 1
        
    # Part 2: A[:, 1:3] * B[1:3, :]
    # A[0,1] -> U[1], A[0,2] -> U[2]
    # A[1,1] -> U[4], A[1,2] -> U[5]
    # B[1,c] -> V[6+c], B[2,c] -> V[12+c]
    # C[0,c] -> W[2c], C[1,c] -> W[2c+1]
    
    # Split into 3 blocks of 2 columns
    for x in range(3):
        c0 = 2 * x
        c1 = 2 * x + 1
        
        # V indices for this block
        v_b1_c0 = 6 + c0
        v_b1_c1 = 6 + c1
        v_b2_c0 = 12 + c0
        v_b2_c1 = 12 + c1
        
        # W indices
        w_c0_0 = 2 * c0
        w_c0_1 = 2 * c0 + 1
        w_c1_0 = 2 * c1
        w_c1_1 = 2 * c1 + 1
        
        # Strassen M1: (a00+a11) * (b00+b11)
        U[1][col] = 1; U[5][col] = 1
        V[v_b1_c0][col] = 1; V[v_b2_c1][col] = 1
        W[w_c0_0][col] = 1
        W[w_c1_1][col] = 1
        col += 1
        
        # Strassen M2: (a10+a11) * b00
        U[4][col] = 1; U[5][col] = 1
        V[v_b1_c0][col] = 1
        W[w_c0_1][col] = 1
        W[w_c1_1][col] = -1
        col += 1
        
        # Strassen M3: a00 * (b01-b11)
        U[1][col] = 1
        V[v_b1_c1][col] = 1
        V[v_b2_c1][col] = -1
        W[w_c1_0][col] = 1
        W[w_c1_1][col] = 1
        col += 1
        
        # Strassen M4: a11 * (b10-b00)
        U[5][col] = 1
        V[v_b2_c0][col] = 1
        V[v_b1_c0][col] = -1
        W[w_c0_0][col] = 1
        W[w_c0_1][col] = 1
        col += 1
        
        # Strassen M5: (a00+a01) * b11
        U[1][col] = 1; U[2][col] = 1
        V[v_b2_c1][col] = 1
        W[w_c0_0][col] = -1
        W[w_c1_0][col] = 1
        col += 1
        
        # Strassen M6: (a10-a00) * (b00+b01)
        U[4][col] = 1; U[1][col] = -1
        V[v_b1_c0][col] = 1
        V[v_b1_c1][col] = 1
        W[w_c1_1][col] = 1
        col += 1
        
        # Strassen M7: (a01-a11) * (b10+b11)
        U[2][col] = 1; U[5][col] = -1
        V[v_b2_c0][col] = 1
        V[v_b2_c1][col] = 1
        W[w_c0_0][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
