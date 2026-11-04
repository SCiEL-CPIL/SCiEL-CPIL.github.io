DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = 5, 5, 5
    
    # Block decomposition:
    # C[0:4, 0:4] = A[0:4, 0:4] * B[0:4, 0:4]  (4x4x4, rank 64)
    # C[0:4, 4]   = A[0:4, 0:5] * B[0:5, 4]    (4x5x1, rank 20)
    # C[4, 0:4]   = A[4, 0:5] * B[0:5, 0:4]    (1x5x4, rank 20)
    # C[4, 4]     = A[4, 0:5] * B[0:5, 4]      (1x5x1, rank 5)
    # Total rank = 64 + 20 + 20 + 5 = 109
    
    rank = 109
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: 4x4x4
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Block 2: 4x5x1 (i<4, j<5, k=4)
    for i in range(4):
        for j in range(5):
            k = 4
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Block 3: 1x5x4 (i=4, j<5, k<4)
    for j in range(5):
        for k in range(4):
            i = 4
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Block 4: 1x5x1 (i=4, j<5, k=4)
    for j in range(5):
        i = 4
        k = 4
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
