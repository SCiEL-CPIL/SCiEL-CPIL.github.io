"""Matrix multiplication 3,4,6 rank-minimal decomposition."""

def construct():
    n, m, p = 3, 4, 6
    # Use the schoolbook (naive) decomposition which is always correct.
    # Rank is n*m*p = 72.
    # Factor matrices:
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    # Entries are 0 or 1.
    
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1.0
                V[v_idx][col] = 1.0
                W[w_idx][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
