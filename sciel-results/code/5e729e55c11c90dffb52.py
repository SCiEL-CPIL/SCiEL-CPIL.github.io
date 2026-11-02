"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """Rank-42 decomposition for 2x2 * 2x12 matrix multiplication.
    
    Uses Strassen's algorithm (rank 7) for each of the 6 independent 2x2 x 2x2 blocks.
    Total rank = 6 * 7 = 42.
    """
    n, m, p = DIMENSIONS  # 2, 2, 12
    num_blocks = p // 2  # 6 blocks of 2 columns
    
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]   # 4 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 24 x 42

    # Strassen terms definition
    # Each term is a tuple (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: list of (row, coeff) for A
    # V_coeffs: list of (row, coeff) for B
    # W_coeffs: list of (row, coeff) for C
    
    # M1 = (A00+A11)(B00+B11)
    m1 = (
        [((0,0), 1), ((1,1), 1)],
        [((0,0), 1), ((1,1), 1)],
        [((0,0), 1), ((1,1), 1)]  # C00+=1, C11+=1
    )
    # M2 = (A10+A11)B00
    m2 = (
        [((1,0), 1), ((1,1), 1)],
        [((0,0), 1)],
        [((1,0), 1), ((1,1), -1)] # C10+=1, C11+=-1
    )
    # M3 = A00(B10-B11)
    m3 = (
        [((0,0), 1)],
        [((1,0), 1), ((1,1), -1)],
        [((0,1), 1), ((1,0), 1)] # C01+=1, C10+=1
    )
    # M4 = A11(B01-B00)
    m4 = (
        [((1,1), 1)],
        [((0,1), 1), ((0,0), -1)],
        [((0,0), -1), ((1,0), 1)] # C00+=-1, C10+=1
    )
    # M5 = (A00+A01)B11
    m5 = (
        [((0,0), 1), ((0,1), 1)],
        [((1,1), 1)],
        [((0,0), 1), ((0,1), 1)] # C00+=1, C01+=1
    )
    # M6 = (A10-A00)(B00+B10)
    m6 = (
        [((1,0), 1), ((0,0), -1)],
        [((0,0), 1), ((1,0), 1)],
        [((1,1), 1)] # C11+=1
    )
    # M7 = (A01-A11)(B01+B11)
    m7 = (
        [((0,1), 1), ((1,1), -1)],
        [((0,1), 1), ((1,1), 1)],
        [((0,0), -1), ((0,1), 1)] # C00+=-1, C01+=1
    )
    
    terms = [m1, m2, m3, m4, m5, m6, m7]

    col = 0
    for b in range(num_blocks):
        base_col = b * 2
        for (u_coeffs, v_coeffs, w_coeffs) in terms:
            for (ai, aj), coeff in u_coeffs:
                u_row = ai * m + aj
                U[u_row][col] = coeff
            for (bj, bk), coeff in v_coeffs:
                # B index: row j, col base_col + bk
                v_row = bj * p + (base_col + bk)
                V[v_row][col] = coeff
            for (ci, ck), coeff in w_coeffs:
                # C index: row i, col base_col + ck
                # Trace convention: W[k*n + i]
                w_row = (base_col + ck) * n + ci
                W[w_row][col] = coeff
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
