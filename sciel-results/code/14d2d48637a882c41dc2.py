"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank-optimal decomposition for M_{2,2,p} with rank 2p
    rank = 2 * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        # Term 2k: A[*][0] * B[0][k] -> C[*][k]
        r0 = 2 * k
        # U: A[0,0] and A[1,0] are selected
        U[0 * m + 0][r0] = 1  # A[0,0]
        U[1 * m + 0][r0] = 1  # A[1,0]
        # V: B[0,k] is selected
        V[0 * p + k][r0] = 1
        # W: C[0,k] and C[1,k] receive contribution
        W[k * n + 0][r0] = 1  # C[0,k]
        W[k * n + 1][r0] = 1  # C[1,k]
        
        # Term 2k+1: A[*][1] * B[1][k] -> C[*][k]
        r1 = 2 * k + 1
        # U: A[0,1] and A[1,1] are selected
        U[0 * m + 1][r1] = 1  # A[0,1]
        U[1 * m + 1][r1] = 1  # A[1,1]
        # V: B[1,k] is selected
        V[1 * p + k][r1] = 1
        # W: C[0,k] and C[1,k] receive contribution
        W[k * n + 0][r1] = 1  # C[0,k]
        W[k * n + 1][r1] = 1  # C[1,k]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
