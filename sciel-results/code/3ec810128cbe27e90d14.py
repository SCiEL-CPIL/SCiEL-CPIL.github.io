<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for idx in range(rank):
        k = idx % p
        j = (idx // p) % m
        i = idx // (p * m)
        U[i * m + j][idx] = 1
        V[j * p + k][idx] = 1
        W[k * n + i][idx] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            U[i * m + j][i] = 1
        for j in range(m):
            for k in range(p):
                V[j * p + k][i] = 1
        for k in range(p):
            W[k * n + i][i] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
