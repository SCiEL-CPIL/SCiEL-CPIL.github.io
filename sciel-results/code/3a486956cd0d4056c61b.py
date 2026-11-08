"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The rank-minimal decomposition for M_{n,m,p} with this specific structure
    # (diagonal in j and k) has rank m*p.
    # U is (n*m, m*p), V is (m*p, m*p), W is (p*n, m*p)
    rank = m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        for k in range(p):
            # For each (j,k) pair, we create a rank-1 term
            # U[i*m+j, col] = 1 for all i
            for i in range(n):
                U[i * m + j][col] = 1
            # V[j*p+k, col] = 1
            V[j * p + k][col] = 1
            # W[k*n+i', col] = 1 for all i'
            for ip in range(n):
                W[k * n + ip][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
