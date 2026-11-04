DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 10  # 5 for i in {0,1}, 5 for i=2
    U = [[0] * rank for _ in range(n * m)]  # (15, 10)
    V = [[0] * rank for _ in range(m * p)]  # (30, 10)
    W = [[0] * rank for _ in range(p * n)]  # (18, 10)
    
    # First group: i in {0,1}, j in 0..4
    for r in range(5):
        j = r
        for i in [0, 1]:
            U[i * m + j][r] = 1
        for k in range(p):
            V[j * p + k][r] = 1
            W[k * n + 0][r] = 1
            W[k * n + 1][r] = 1
    
    # Second group: i=2, j in 0..4
    for r in range(5):
        j = r
        r2 = 5 + r
        U[2 * m + j][r2] = 1
        for k in range(p):
            V[j * p + k][r2] = 1
            W[k * n + 2][r2] = 1
    
    return {"U": U, "V": V, "W": W}
