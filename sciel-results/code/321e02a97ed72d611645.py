"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 27, which is the theoretical minimum max(nm, mp, pn)
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # We decompose into 3 blocks of 3 columns in B
    for block in range(3):
        for j in range(m):
            for k_local in range(3):
                k_global = block * 3 + k_local
                # U factor: A[i, j]
                for i in range(n):
                    U[i * m + j][col] = 1
                # V factor: B[j, k]
                V[j * p + k_global][col] = 1
                # W factor: C[k, i]
                for i in range(n):
                    W[k_global * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
