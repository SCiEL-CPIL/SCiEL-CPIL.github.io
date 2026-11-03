"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Lower bound for rank of M_{n,m,p} is max(n, m, p)
    R = max(n, m, p)
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # U[i*m + j, r] = 1 if j == r
    for i in range(n):
        for j in range(m):
            if j < R:
                U[i * m + j][j] = 1
            else:
                # If m > R, we need a different strategy, but here m=4, R=4
                # So j is always < R
                pass
                
    # V[j*p + k, r] = 1 if j == r
    for j in range(m):
        for k in range(p):
            if j < R:
                V[j * p + k][j] = 1
            else:
                pass
                
    # W[k*n + i, r] = 1 for all r
    for k in range(p):
        for i in range(n):
            for r in range(R):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
