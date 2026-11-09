"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p.
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for term c.
    # Index c corresponds to the triplet (i, j, k).
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Populate the decomposition
    # c = i * m * p + j * p + k
    # u_idx = i * m + j
    # v_idx = j * p + k
    # w_idx = k * n + i
    for c in range(rank):
        # Decode c into i, j, k
        i = c // (m * p)
        rem = c % (m * p)
        j = rem // p
        k = rem % p
        
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        U[u_idx][c] = 1.0
        V[v_idx][c] = 1.0
        W[w_idx][c] = 1.0
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
