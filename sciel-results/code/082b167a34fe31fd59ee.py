import numpy as np

def construct():
    n, m, p = 3, 3, 8
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[0:2, 0:2] * B[0:2, 0:8]
    # 4 Strassen blocks for k pairs (0,1), (2,3), (4,5), (6,7)
    for k_start in [0, 2, 4, 6]:
        k0 = k_start
        k1 = k_start + 1
        
        # Strassen M0 = (A00 + A11)(B00 + B11)
        # U[0,0]=1, U[1,1]=1
        # V[0,k0]=1, V[1,k1]=1
        # W[k0,0]=1, W[k1,1]=1
        U[0 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M1 = (A00 + A11)(B10 - B00)
        # U[0,0]=1, U[1,1]=1
        # V[1,k0]=1, V[0,k0]=-1
        # W[k0,0]=1, W[k0,1]=1
        U[0 * m + 0][col] = 1
        U[1 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        V[0 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # M2 = A10(B01 - B11)
        # U[1,0]=1
        # V[0,k1]=1, V[1,k1]=-1
        # W[k1,0]=1, W[k1,1]=-1
        U[1 * m + 0][col] = 1
        V[0 * p + k1][col] = 1
        V[1 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # M3 = A01(B10 + B11)
        # U[0,1]=1
        # V[1,k0]=1, V[1,k1]=1
        # W[k0,0]=1, W[k1,0]=1
        U[0 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # M4 = (A00 - A10)(B00 + B01)
        # U[0,0]=1, U[1,0]=-1
        # V[0,k0]=1, V[0,k1]=1
        # W[k0,0]=1, W[k1,0]=1
        U[0 * m + 0][col] = 1
        U[1 * m + 0][col] = -1
        V[0 * p + k0][col] = 1
        V[0 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # M5 = (A11 - A01)(B10 + B11)
        # U[1,1]=1, U[0,1]=-1
        # V[1,k0]=1, V[1,k1]=1
        # W[k0,1]=1, W[k1,1]=1
        U[1 * m + 1][col] = 1
        U[0 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # M6 = (A00 + A01)(B00 - B10)
        # U[0,0]=1, U[0,1]=1
        # V[0,k0]=1, V[1,k0]=-1
        # W[k0,0]=1, W[k0,1]=-1
        U[0 * m + 0][col] = 1
        U[0 * m + 1][col] = 1
        V[0 * p + k0][col] = 1
        V[1 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = -1
        col += 1

    # Part 2: A[0:2, 2] * B[2, 0:8]
    # Rank 1
    U[0 * m + 2][col] = 1
    U[1 * m + 2][col] = 1
    for k in range(p):
        V[2 * p + k][col] = 1
        W[k * n + 0][col] = 1
        W[k * n + 1][col] = 1
    col += 1

    # Part 3: A[2, 0:3] * B[0:3, 0:8]
    # Schoolbook 24 terms
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
