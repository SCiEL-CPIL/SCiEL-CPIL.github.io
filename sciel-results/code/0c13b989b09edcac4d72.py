# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Optimal rank for 2x2x7 is 28.
    # We construct the decomposition by mapping each (i, j, k) tuple to a unique column.
    rank = n * m * p
    
    # Pre-allocate matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Populate the matrices
    # The bilinear forms are A[i,j] * B[j,k]
    # The output is C[i,k]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
