"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j0 in range(m):
        for k0 in range(p):
            c = j0 * p + k0
            # U[i*m + j, c] = 1 if j == j0, else 0 (for all i)
            for i in range(n):
                U[i * m + j0][c] = 1
            # V[j'*p + k, c] = 1 if j' == j0 and k == k0, else 0
            V[j0 * p + k0][c] = 1
            # W[k'*n + i', c] = 1 if k' == k0, else 0 (for all i')
            for i_prime in range(n):
                W[k0 * n + i_prime][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
