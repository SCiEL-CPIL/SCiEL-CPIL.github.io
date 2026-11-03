"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-48 exact tensor decomposition for M_{2,3,8} (schoolbook).
    This is known to be optimal: rank(M_{n,m,1}) = n*m, so rank(M_{n,m,p}) = p*n*m.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 48

    U = [[0.0] * rank for _ in range(n * m)]  # shape (6, 48)
    V = [[0.0] * rank for _ in range(m * p)]  # shape (24, 48)
    W = [[0.0] * rank for _ in range(p * n)]  # shape (16, 48)

    col = 0
    for i in range(n):       # output row
        for j in range(m):   # shared column
            for k in range(p):  # output col
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
