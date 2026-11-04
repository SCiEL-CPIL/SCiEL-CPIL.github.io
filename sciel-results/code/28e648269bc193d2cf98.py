import numpy as np

def construct():
    n, m, p = 4, 4, 5
    
    # Rank-7 Strassen for 2x2x2
    def strassen_2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # c=0: M1 = (A11+A22)(B11+B22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # c=1: M2 = (A21+A22)B11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # c=2: M3 = A11(B12-B22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # c=3: M4 = A22(B21-B11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # c=4: M5 = (A11+A12)B22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # c=5: M6 = (A21-A11)(B11+B12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # c=6: M7 = (A12-A22)(B21+B22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = -1
        
        return U, V, W
    
    U4, V4, W4 = strassen_2x2()
    
    # Build 4x4x4 using tensor product: rank 49
    rank_444 = 49
    U_444 = [[0]*rank_444 for _ in range(16)]
    V_444 = [[0]*rank_444 for _ in range(16)]
    W_444 = [[0]*rank_444 for _ in range(16)]
    
    for co in range(7):
        for ci in range(7):
            col = co * 7 + ci
            for i in range(4):
                for j in range(4):
                    row = i * 4 + j
                    outer = (i // 2) * 2 + (j // 2)
                    inner = (i % 2) * 2 + (j % 2)
                    U_444[row][col] = U4[outer][co] * U4[inner][ci]
            
            for j in range(4):
                for k in range(4):
                    row = j * 4 + k
                    outer = (j // 2) * 2 + (k // 2)
                    inner = (j % 2) * 2 + (k % 2)
                    V_444[row][col] = V4[outer][co] * V4[inner][ci]
            
            for k in range(4):
                for i in range(4):
                    row = k * 4 + i
                    outer = (k // 2) * 2 + (i // 2)
                    inner = (k % 2) * 2 + (i % 2)
                    W_444[row][col] = W4[outer][co] * W4[inner][ci]
    
    # Full problem: rank = 49 + 16 = 65
    rank = 65
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(20)]
    W = [[0]*rank for _ in range(20)]
    
    # Copy 4x4x4 part
    for r in range(16):
        for c in range(49):
            U[r][c] = U_444[r][c]
            V[r][c] = V_444[r][c]
            W[r][c] = W_444[r][c]
    
    # Add 4x4x1 schoolbook for 5th column
    col = 49
    for i in range(4):
        for j in range(4):
            U[i*4 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[4*4 + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
