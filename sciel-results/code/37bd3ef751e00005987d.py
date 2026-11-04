"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Rank-minimal decomposition exploiting the structure of C_i,k = sum_j A_i,j B_j,k
    # The tensor decomposes into the direct sum of n independent 1xmxp multiplications.
    # Rank of 1xmxp is m. Total rank is n*m.
    n, m, p = DIMENSIONS
    rank = n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            # U[i,j, r] = 1
            U[i * m + j][r] = 1
            # V[j,k, r] = 1 for all k
            for k in range(p):
                V[j * p + k][r] = 1
            # W[k,i, r] = 1 for all k
            for k in range(p):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
