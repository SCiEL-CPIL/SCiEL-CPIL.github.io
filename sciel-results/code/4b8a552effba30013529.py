"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=5, p=5
    # Rank-2 decomposition exploiting the all-ones tensor structure
    rank = 2
    U = [[0] * rank for _ in range(n * m)]  # (10, 2)
    V = [[0] * rank for _ in range(m * p)]  # (25, 2)
    W = [[0] * rank for _ in range(p * n)]  # (10, 2)
    
    # For r=0: handle i=0 output row
    # U[i*m+j, 0] = 1 if i==0 else 0
    # V[j*p+k, 0] = 1 for all j,k
    # W[k*n+i, 0] = 1 if i==0 else 0
    for j in range(m):
        U[0 * m + j][0] = 1  # i=0, all j
    for j in range(m):
        for k in range(p):
            V[j * p + k][0] = 1
    for k in range(p):
        W[k * n + 0][0] = 1  # i=0, all k
    
    # For r=1: handle i=1 output row
    # U[i*m+j, 1] = 1 if i==1 else 0
    # V[j*p+k, 1] = 1 for all j,k
    # W[k*n+i, 1] = 1 if i==1 else 0
    for j in range(m):
        U[1 * m + j][1] = 1  # i=1, all j
    for j in range(m):
        for k in range(p):
            V[j * p + k][1] = 1
    for k in range(p):
        W[k * n + 1][1] = 1  # i=1, all k
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
