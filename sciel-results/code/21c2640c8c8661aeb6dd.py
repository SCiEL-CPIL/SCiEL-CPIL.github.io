"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m  # Trivial decomposition rank is m
    
    # U shape (n*m, rank)
    # V shape (m*p, rank)
    # W shape (p*n, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        # r corresponds to the summation index j in the matrix multiplication
        # A indices: i in {0, 1}, j = r
        # U index: i * m + j
        for i in range(n):
            idx_u = i * m + r
            U[idx_u][r] = 1
        
        # B indices: j = r, k in {0, ..., p-1}
        # V index: j * p + k
        for k in range(p):
            idx_v = r * p + k
            V[idx_v][r] = 1
        
        # C indices: i in {0, 1}, k in {0, ..., p-1}
        # W index: k * n + i (Trace convention W[k*n+i])
        for k in range(p):
            for i in range(n):
                idx_w = k * n + i
                W[idx_w][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
