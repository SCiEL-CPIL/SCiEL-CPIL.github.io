DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 27 Fallback (Schoolbook)
    # This is the guaranteed valid solution.
    U = [[0] * 27 for _ in range(n * m)]
    V = [[0] * 27 for _ in range(m * p)]
    W = [[0] * 27 for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
