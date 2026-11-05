"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 8 decomposition
    # Based on the structure of 1xm x px1 being rank m, and here we have 2 rows.
    # We use 4 terms for row 1 and 4 terms for row 2.
    rank = 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Row 1 (i=0) uses r=0..3
    # Row 2 (i=1) uses r=4..7
    for r in range(4):
        # U for row 1, column j=r
        # Index in U: i*m + j = 0*m + r = r
        U[r][r] = 1
        # V for column j=r, all k
        # Index in V: j*p + k = r*p + k
        for k in range(p):
            V[r * p + k][r] = 1
        # W for row 1 (i=0), all k
        # Index in W: k*n + i = k*2 + 0 = 2k
        for k in range(p):
            W[k * n + 0][r] = 1
            
    for r in range(4):
        # U for row 2, column j=r
        # Index in U: i*m + j = 1*m + r = 4 + r
        U[m + r][4 + r] = 1
        # V for column j=r, all k
        # Index in V: j*p + k = r*p + k
        for k in range(p):
            V[r * p + k][4 + r] = 1
        # W for row 2 (i=1), all k
        # Index in W: k*n + i = k*2 + 1 = 2k + 1
        for k in range(p):
            W[k * n + 1][4 + r] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
