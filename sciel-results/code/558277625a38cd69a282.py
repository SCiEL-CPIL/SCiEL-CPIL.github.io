DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 61
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. Top-left 4x4 block (i,j,k in 0..3)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # 2. Right column (i in 0..3, k=4)
    # C[i,4] = sum_{j=0}^3 A[i,j] B[j,4] + A[i,4] B[4,4]
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
        # A[i,4] B[4,4]
        U[i * m + 4][col] = 1
        V[4 * p + 4][col] = 1
        W[4 * n + i][col] = 1
        col += 1

    # 3. Bottom row (i=4, k in 0..3)
    # C[4,k] = sum_{j=0}^3 A[4,j] B[j,k] + A[4,4] B[4,k]
    for k in range(4):
        for j in range(4):
            U[4 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 4][col] = 1
            col += 1
        # A[4,4] B[4,k]
        U[4 * m + 4][col] = 1
        V[4 * p + k][col] = 1
        W[k * n + 4][col] = 1
        col += 1

    # 4. Bottom-right (i=4, k=4)
    # C[4,4] = sum_{j=0}^3 A[4,j] B[j,4] + A[4,4] B[4,4]
    for j in range(4):
        U[4 * m + j][col] = 1
        V[j * p + 4][col] = 1
        W[4 * n + 4][col] = 1
        col += 1
    # A[4,4] B[4,4]
    U[4 * m + 4][col] = 1
    V[4 * p + 4][col] = 1
    W[4 * n + 4][col] = 1
    col += 1

    assert col == rank
    return {"U": U, "V": V, "W": W}
