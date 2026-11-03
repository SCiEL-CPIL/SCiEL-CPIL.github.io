DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Use block decomposition: split middle dimension 4 into two 2s
    # A = [A0 A1] (2x2 each), B = [B0; B1] (2x4 each)
    # C = A0*B0 + A1*B1
    # Each 2x2 * 2x4 has schoolbook rank 16, total 32
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]  # 2*4 = 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 4*8 = 32 rows
    W = [[0] * rank for _ in range(p * n)]  # 8*2 = 16 rows
    
    col = 0
    # A0*B0: A0 is rows 0-1, cols 0-1 of A; B0 is rows 0-1 of B
    for i in range(2):
        for j in range(2):
            for k in range(4):
                U[i * 4 + j][col] = 1
                V[j * 8 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1
    
    # A1*B1: A1 is rows 0-1, cols 2-3 of A; B1 is rows 2-3 of B
    for i in range(2):
        for j in range(2):
            for k in range(4):
                U[i * 4 + (j + 2)][col] = 1
                V[(j + 2) * 8 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
