import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    # Target: Rank 36 decomposition for 2x4x5
    
    n, m, p = 2, 4, 5
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    # 1. Define Strassen 2x2x2 decomposition
    # U (4x7), V (4x7), W (4x7)
    U_s = [
        [1, 0, 1, 0, 1, -1, 0], # i=0, j=0
        [0, 0, 0, 0, 1, 0, 1],  # i=0, j=1
        [0, 1, 0, 0, 0, 1, 0],  # i=1, j=0
        [1, 1, 0, 1, 0, 0, -1]  # i=1, j=1
    ]
    V_s = [
        [1, 1, 0, -1, 0, 1, 0], # j=0, k=0
        [0, 0, 1, 0, 0, 1, 0],  # j=0, k=1
        [0, 0, 0, 1, 0, 0, 1],  # j=1, k=0
        [1, 0, -1, 0, 1, 0, 1]  # j=1, k=1
    ]
    W_s = [
        [1, 0, 0, 1, -1, 0, 1], # k=0, i=0
        [0, 0, 1, 0, 1, 0, 0],  # k=0, i=1
        [0, 1, 0, 1, 0, 0, 0],  # k=1, i=0
        [1, -1, 1, 0, 0, 1, 0]  # k=1, i=1
    ]
    
    # 2. Build 2x2x4 decomposition (Rank 14)
    # Split B (2x4) into B1 (2x2) and B2 (2x2)
    # U_14 (4x14), V_14 (8x14), W_14 (8x14)
    
    U_14 = [[0]*14 for _ in range(4)]
    V_14 = [[0]*14 for _ in range(8)]
    W_14 = [[0]*14 for _ in range(8)]
    
    # Copy 1: k=0,1
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U_14[2*i+j][r] = U_s[2*i+j][r]
        for j in range(2):
            for k in range(2):
                V_14[2*j+k][r] = V_s[2*j+k][r]
        for k in range(2):
            for i in range(2):
                W_14[2*k+i][r] = W_s[2*k+i][r]
                
    # Copy 2: k=2,3
    for r in range(7):
        r2 = r + 7
        for i in range(2):
            for j in range(2):
                U_14[2*i+j][r2] = U_s[2*i+j][r]
        for j in range(2):
            for k in range(2):
                # Local k is 0,1. Global k is 2,3.
                # Global V row: 2*j + (k+2) = 2*j + k + 2
                V_14[2*j + k + 2][r2] = V_s[2*j+k][r]
        for k in range(2):
            for i in range(2):
                # Local k is 0,1. Global k is 2,3.
                # Global W row: 2*(k+2) + i = 2*k + i + 4
                W_14[2*k + i + 4][r2] = W_s[2*k+i][r]
                
    # 3. Build 2x2x5 decomposition (Rank 18)
    # Split B (2x5) into B1 (2x4) and B2 (2x1)
    # U_18 (4x18), V_18 (10x18), W_18 (10x18)
    
    # First part: 2x2x4 (Rank 14)
    # Second part: 2x2x1 (Rank 4) - Schoolbook
    
    U_18 = [[0]*18 for _ in range(4)]
    V_18 = [[0]*18 for _ in range(10)]
    W_18 = [[0]*18 for _ in range(10)]
    
    # Fill first 14 cols from U_14, V_14, W_14
    for r in range(14):
        for i in range(2):
            for j in range(2):
                U_18[2*i+j][r] = U_14[2*i+j][r]
        for j in range(2):
            for k in range(4):
                V_18[2*j+k][r] = V_14[2*j+k][r]
        for k in range(4):
            for i in range(2):
                W_18[2*k+i][r] = W_14[2*k+i][r]
                
    # Fill last 4 cols with Schoolbook 2x2x1
    # Indices: i in 0,1; j in 0,1; k=0 (local)
    # Global k for this part is 4.
    for r_local in range(4):
        r = r_local + 14
        i = r_local // 2
        j = r_local % 2
        
        # U: depends on i, j
        U_18[2*i+j][r] = 1
        
        # V: depends on j, k_local(0)
        # Global k = 4. Row = 2*j + 4
        V_18[2*j + 4][r] = 1
        
        # W: depends on k_global(4), i
        # Row = 2*4 + i = 8 + i
        W_18[8 + i][r] = 1
        
    # 4. Build 2x4x5 decomposition (Rank 36)
    # Split A (2x4) into A_L (2x2) and A_R (2x2)
    # Split B (4x5) into B_T (2x5) and B_B (2x5)
    # C = A_L B_T + A_R B_B
    
    # Copy 1: A_L B_T (uses U_18, V_18, W_18)
    # j_local in 0,1 -> j_global in 0,1
    # k in 0..4 -> k_global in 0..4
    # i in 0,1 -> i_global in 0,1
    
    # Copy 2: A_R B_B (uses U_18, V_18, W_18)
    # j_local in 0,1 -> j_global in 2,3
    # k in 0..4 -> k_global in 0..4
    # i in 0,1 -> i_global in 0,1
    
    for r_local in range(18):
        # Copy 1 (cols 0-17)
        r1 = r_local
        for i in range(2):
            for j in range(2):
                U[2*i+j][r1] = U_18[2*i+j][r_local]
        for j in range(2):
            for k in range(5):
                V[2*j+k][r1] = V_18[2*j+k][r_local]
        for k in range(5):
            for i in range(2):
                W[2*k+i][r1] = W_18[2*k+i][r_local]
                
        # Copy 2 (cols 18-35)
        r2 = r_local + 18
        for i in range(2):
            for j in range(2):
                # j_global = j + 2
                U[2*i + (j+2)][r2] = U_18[2*i+j][r_local]
        for j in range(2):
            for k in range(5):
                # j_global = j + 2
                V[2*(j+2) + k][r2] = V_18[2*j+k][r_local]
        for k in range(5):
            for i in range(2):
                W[2*k+i][r2] = W_18[2*k+i][r_local]
                
    return {"U": U, "V": V, "W": W}
