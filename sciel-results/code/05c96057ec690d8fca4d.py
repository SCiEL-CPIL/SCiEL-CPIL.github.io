import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    assert n == 6 and m == 6 and p == 6

    # Schoolbook decomposition
    # Rank = n * m * p = 216
    rank = n * m * p
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # r corresponds to (j, i, k)
    # U[i, j, r] = 1 if r matches (j, i, k)
    # V[j, k, r] = 1 if r matches (j, i, k)
    # W[k, i, r] = 1 if r matches (j, i, k)
    
    # Index mapping:
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    r = 0
    for j in range(m):
        for i in range(n):
            for k in range(p):
                # U
                u_idx = i * m + j
                U[u_idx][r] = 1.0
                
                # V
                v_idx = j * p + k
                V[v_idx][r] = 1.0
                
                # W
                w_idx = k * n + i
                W[w_idx][r] = 1.0
                
                r += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
