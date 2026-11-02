import numpy as np

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 3 * 7 (Strassen for 2x2x2) + 4 (Schoolbook for 2x2x1) = 25
    rank = 25
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen 2x2x2 coefficients
    # U, V, W are 4x7 matrices for the local block
    # Local indices: A: 00,01,10,11 -> 0,1,2,3
    #                B: 00,01,10,11 -> 0,1,2,3
    #                C: 00,01,10,11 -> 0,1,2,3
    
    U_strassen = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, -1], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    V_strassen = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    W_strassen = [
        [1, 0, 0, 1], # M1 -> C00, C11
        [0, 0, 1, 0], # M2 -> C10
        [0, 1, 0, 0], # M3 -> C01
        [1, 0, 1, 0], # M4 -> C00, C10
        [-1, 1, 0, 0], # M5 -> C00, C01
        [0, 0, 0, 1], # M6 -> C11
        [1, 0, 0, 0]  # M7 -> C00
    ]
    
    # Define Schoolbook 2x2x1 coefficients
    # U, V, W are 4x4, 2x4, 2x4
    # Local A: 0,1,2,3
    # Local B: 0,1 (col vector)
    # Local C: 0,1 (col vector)
    
    U_schoolbook = [
        [1, 0, 0, 0], # M1
        [0, 1, 0, 0], # M2
        [0, 0, 1, 0], # M3
        [0, 0, 0, 1]  # M4
    ]
    
    V_schoolbook = [
        [1, 0, 1, 0], # B0 for M1, M3
        [0, 1, 0, 1]  # B1 for M2, M4
    ]
    
    W_schoolbook = [
        [1, 1, 0, 0], # C0 for M1, M2
        [0, 0, 1, 1]  # C1 for M3, M4
    ]
    
    # Fill U
    # Blocks 0, 1, 2 use Strassen
    for b in range(3):
        for r in range(7):
            col = b * 7 + r
            for i in range(4):
                U[i][col] = U_strassen[i][r]
    
    # Block 3 uses Schoolbook
    for r in range(4):
        col = 21 + r
        for i in range(4):
            U[i][col] = U_schoolbook[i][r]
            
    # Fill V
    # V has shape (m*p, rank) = (14, 25)
    # Row index j*p + k
    for b in range(3):
        for r in range(7):
            col = b * 7 + r
            for j in range(2):
                for k_loc in range(2):
                    k_glob = b * 2 + k_loc
                    row = j * p + k_glob
                    V[row][col] = V_strassen[j * 2 + k_loc][r]
                    
    for r in range(4):
        col = 21 + r
        for j in range(2):
            k_glob = 6
            row = j * p + k_glob
            V[row][col] = V_schoolbook[j][r]
            
    # Fill W
    # W has shape (p*n, rank) = (14, 25)
    # Row index k*n + i
    for b in range(3):
        for r in range(7):
            col = b * 7 + r
            for k_loc in range(2):
                k_glob = b * 2 + k_loc
                for i in range(2):
                    row = k_glob * n + i
                    W[row][col] = W_strassen[k_loc * 2 + i][r]
                    
    for r in range(4):
        col = 21 + r
        k_glob = 6
        for i in range(2):
            row = k_glob * n + i
            W[row][col] = W_schoolbook[i][r]
            
    return {"U": U, "V": V, "W": W}
