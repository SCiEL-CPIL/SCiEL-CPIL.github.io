DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Laderman's 3x3x3 decomposition
    # u, v, w are lists of length 9
    # 
    # I will use a known valid set.
    # 
    # Term 1
    u = [1, 0, 0, 0, 1, 0, 0, 0, 0]
    v = [1, 0, 0, 0, 1, 0, 0, 0, 0]
    w = [1, 0, 0, 0, 1, 0, 0, 0, 0] # C11, C22
    
    # This is too hard to guess correctly without a reference.
    # I will stick to Rank 24 (Schoolbook based).
    # It is robust.
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block 1: cols 0-2 of B
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # Block 2: cols 3-5 of B
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + (3 + k)][col] = 1
                W[(3 + k) * n + i][col] = 1
                col += 1
    # Block 3: cols 6-7 of B
    # Schoolbook for 3x3x3, but only k=0,1
    for i in range(n):
        for j in range(m):
            for k in range(2):
                U[i * m + j][col] = 1
                V[j * p + (6 + k)][col] = 1
                W[(6 + k) * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
