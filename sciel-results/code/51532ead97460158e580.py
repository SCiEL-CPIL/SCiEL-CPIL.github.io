def _schoolbook_block(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    # Split p into 3, 3, 2
    p1, p2, p3 = 3, 3, 2
    
    U1, V1, W1, r1 = _schoolbook_block(n, m, p1)
    U2, V2, W2, r2 = _schoolbook_block(n, m, p2)
    U3, V3, W3, r3 = _schoolbook_block(n, m, p3)
    
    total_rank = r1 + r2 + r3
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill U
    for r in range(total_rank):
        if r < r1:
            for i in range(n * m):
                U[i][r] = U1[i][r]
        elif r < r1 + r2:
            for i in range(n * m):
                U[i][r] = U2[i][r - r1]
        else:
            for i in range(n * m):
                U[i][r] = U3[i][r - r1 - r2]
                
    # Fill V
    # V1 is m x p1. V2 is m x p2. V3 is m x p3.
    # V is m x p.
    for r in range(total_rank):
        if r < r1:
            for j in range(m):
                for k in range(p1):
                    V[j * p + k][r] = V1[j * p1 + k][r]
        elif r < r1 + r2:
            for j in range(m):
                for k in range(p2):
                    V[j * p + (p1 + k)][r] = V2[j * p2 + k][r - r1]
        else:
            for j in range(m):
                for k in range(p3):
                    V[j * p + (p1 + p2 + k)][r] = V3[j * p3 + k][r - r1 - r2]
                    
    # Fill W
    # W1 is p1 x n. W2 is p2 x n. W3 is p3 x n.
    # W is p x n.
    for r in range(total_rank):
        if r < r1:
            for k in range(p1):
                for i in range(n):
                    W[k * n + i][r] = W1[k * n + i][r]
        elif r < r1 + r2:
            for k in range(p2):
                for i in range(n):
                    W[(p1 + k) * n + i][r] = W2[k * n + i][r - r1]
        else:
            for k in range(p3):
                for i in range(n):
                    W[(p1 + p2 + k) * n + i][r] = W3[k * n + i][r - r1 - r2]
                    
    return {"U": U, "V": V, "W": W}
