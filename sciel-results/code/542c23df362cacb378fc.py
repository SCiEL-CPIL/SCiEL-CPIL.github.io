"""Program for 3x4x7 matrix multiplication tensor decomposition."""

def construct():
    n, m, p = 3, 4, 7
    rank = 48
    
    U = [[0] * rank for _ in range(n * m)]  # 12 x 48
    V = [[0] * rank for _ in range(m * p)]  # 28 x 48
    W = [[0] * rank for _ in range(p * n)]  # 21 x 48
    
    col = 0
    
    # Block decomposition:
    # A (3x4) = [A1 (3x3) | A2 (3x1)]
    # B (4x7) = [B1 (3x7); B2 (1x7)]
    # C = A1*B1 + A2*B2
    
    # A1*B1: Split B1 into B1a (3x3), B1b (3x3), B1c (3x1)
    
    # Part 1: A1 (3x3) * B1a (3x3) -> C1a (3x3), schoolbook rank 9
    # A1 = A[0:3, 0:3], B1a = B[0:3, 0:3], C1a = C[0:3, 0:3]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 4 + j][col] = 1
                V[j * 7 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    # col = 9
    
    # Part 2: A1 (3x3) * B1b (3x3) -> C1b (3x3), schoolbook rank 9
    # B1b = B[0:3, 3:6], C1b = C[3:6, 0:3]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 4 + j][col] = 1
                V[j * 7 + (3 + k)][col] = 1
                W[(3 + k) * 3 + i][col] = 1
                col += 1
    # col = 18
    
    # Part 3: A1 (3x3) * B1c (3x1) -> C1c (3x1), schoolbook rank 9
    # B1c = B[0:3, 6:7], C1c = C[6:7, 0:3]
    for i in range(3):
        for j in range(3):
            U[i * 4 + j][col] = 1
            V[j * 7 + 6][col] = 1
            W[6 * 3 + i][col] = 1
            col += 1
    # col = 27
    
    # Part 4: A2 (3x1) * B2 (1x7) -> C2 (3x7), schoolbook rank 21
    # A2 = A[0:3, 3:4], B2 = B[3:4, 0:7], C2 = C[0:7, 0:3]
    for i in range(3):
        for k in range(7):
            U[i * 4 + 3][col] = 1
            V[3 * 7 + k][col] = 1
            W[k * 3 + i][col] = 1
            col += 1
    # col = 48
    
    return {"U": U, "V": V, "W": W}
