"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Naive decomposition: Rank = n * m * p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Iterate over all i, j, k
    # t is the rank index
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                u_idx = i * m + j
                U[u_idx][t] = 1
                
                # V index: j*p + k
                v_idx = j * p + k
                V[v_idx][t] = 1
                
                # W index: k*n + i
                w_idx = k * n + i
                W[w_idx][t] = 1
                
                t += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
