import numpy as np

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 31 decomposition
    # 7 terms for Strassen on first 2 columns
    # 4 terms for each of the remaining 6 columns
    rank = 7 + 4 * (p - 2)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen for columns 0, 1 (indices 0, 1 in 0-based)
    # B indices: j*p + k
    # k=0, 1
    
    # M1 = (A11 + A22)(B10 + B21)
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[9][col] = 1
    W[1][col] = 1; W[10][col] = 1 # C10, C21? No.
    # C10 = M5 + M1 - M3 - M4
    # C11 = M4 + M5
    # C20 = M2 + M3
    # C21 = M1 - M2 + M6 - M7
    
    # Let's map W indices: W[k*n + i]
    # C[0,0] -> idx 0
    # C[1,0] -> idx 1
    # C[0,1] -> idx 2
    # C[1,1] -> idx 3
    
    # M1
    U[0][col] = 1; U[3][col] = 1
    V[0][col] = 1; V[9][col] = 1
    W[0][col] = 1; W[1][col] = 1; W[2][col] = 1; W[3][col] = 1
    # Wait, W is the output vector.
    # M1 contributes to C10, C21, C11, C20?
    # C10 = ... + M1 ...
    # C11 = ...
    # C20 = ...
    # C21 = ... + M1 ...
    # So W[0]=1, W[1]=1, W[2]=1, W[3]=1?
    # C10 (idx 0), C20 (idx 1), C11 (idx 2), C21 (idx 3).
    # Yes.
    col += 1
    
    # M2 = (A21 + A22) B10
    U[2][col] = 1; U[3][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = -1 # C20, C21
    col += 1
    
    # M3 = A11 (B20 - B21)
    U[0][col] = 1
    V[8][col] = 1; V[9][col] = -1
    W[0][col] = -1; W[1][col] = 1 # C10, C20
    col += 1
    
    # M4 = A22 (B11 - B10)
    U[3][col] = 1
    V[1][col] = 1; V[0][col] = -1
    W[0][col] = -1; W[2][col] = 1 # C10, C11
    col += 1
    
    # M5 = (A11 + A12) B21
    U[0][col] = 1; U[1][col] = 1
    V[9][col] = 1
    W[0][col] = 1; W[2][col] = 1 # C10, C11
    col += 1
    
    # M6 = (A21 - A11) (B10 + B11)
    U[2][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1 # C21
    col += 1
    
    # M7 = (A12 - A22) (B20 + B21)
    U[1][col] = 1; U[3][col] = -1
    V[8][col] = 1; V[9][col] = 1
    W[3][col] = -1 # C21
    col += 1
    
    # Remaining columns k=2..7
    for k in range(2, p):
        # C1k = A11 B1k + A12 B2k
        # C2k = A21 B1k + A22 B2k
        
        # T1: A11 * B1k -> C1k
        U[0][col] = 1
        V[k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        
        # T2: A12 * B2k -> C1k
        U[1][col] = 1
        V[8+k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        
        # T3: A21 * B1k -> C2k
        U[2][col] = 1
        V[k][col] = 1
        W[k*n + 1][col] = 1
        col += 1
        
        # T4: A22 * B2k -> C2k
        U[3][col] = 1
        V[8+k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
