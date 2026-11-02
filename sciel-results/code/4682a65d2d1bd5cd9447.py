"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 10
    num_blocks = p // 2  # 5
    rank = num_blocks * 7  # 35

    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35

    # Strassen factors for one 2x2 x 2x2 block, indexed by factor f in 0..6
    # U[i*m+j][f], i,j in 0..1
    U_factors = [
        [1, 0, 0, 0, 1, -1, 0],  # (i=0,j=0) A00
        [0, 0, 0, 0, 1, 0, 1],   # (i=0,j=1) A01
        [0, 1, 0, 0, 0, 1, 0],   # (i=1,j=0) A10
        [1, 1, 0, 1, 0, 0, -1],  # (i=1,j=1) A11
    ]
    # V[j*2+l][f], j,l in 0..1 (local)
    V_factors = [
        [1, 1, 0, -1, 0, 1, 0],  # (j=0,l=0) B00
        [0, 0, 0, 1, 0, 0, 1],   # (j=0,l=1) B01
        [1, 0, 1, 0, 0, 1, 0],   # (j=1,l=0) B10
        [1, 0, -1, 0, 1, 0, 1],  # (j=1,l=1) B11
    ]
    # W[l*2+i][f], l,i in 0..1 (local)
    W_factors = [
        [1, 0, 0, 1, -1, 0, 1],  # (l=0,i=0) C00
        [0, 1, 0, 1, 0, 0, 0],   # (l=0,i=1) C10
        [0, 0, 1, 0, 1, 0, 0],   # (l=1,i=0) C01
        [1, -1, 1, 0, 0, 1, 0],  # (l=1,i=1) C11
    ]

    for b in range(num_blocks):
        base = b * 7
        for f in range(7):
            col = base + f
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] = U_factors[i * 2 + j][f]
            for j in range(2):
                for l_local in range(2):
                    l_global = 2 * b + l_local
                    V[j * p + l_global][col] = V_factors[j * 2 + l_local][f]
            for l_local in range(2):
                l_global = 2 * b + l_local
                for i in range(2):
                    W[l_global * n + i][col] = W_factors[l_local * 2 + i][f]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
