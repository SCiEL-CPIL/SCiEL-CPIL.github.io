"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-22 decomposition of the (2,3,4) matrix multiplication tensor.
    """
    n, m, p = DIMENSIONS  # 2, 3, 4
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen-like 7-term pattern for 2x2x2 multiplication.
    # U indices: 0=A00, 1=A01, 2=A10, 3=A11
    # V indices: 0=B00, 1=B01, 2=B10, 3=B11
    # W indices: 0=C00, 1=C01, 2=C10, 3=C11
    strassen = [
        # (u_idx, val, v_idx, val, w_idx, val)
        (0, 1, 0, 1, 0, 1),  # M1
        (3, 1, 3, 1, 0, 1),  # M1
        (1, 1, 1, -1, 1, 1), # M1
        (1, 1, 1, -1, 3, 1), # M1
        (1, 1, 1, -1, 2, -1),# M1
        (2, 1, 2, 1, 2, 1),  # M2
        (3, 1, 3, 1, 2, 1),  # M2
        (0, 1, 1, 1, 1, 1),  # M3
        (1, 1, 1, 1, 1, 1),  # M3
        (3, 1, 0, -1, 3, 1), # M4
        (2, 1, 2, 1, 3, 1),  # M4
        (3, 1, 2, 1, 3, 1),  # M4
        (3, 1, 2, 1, 0, -1), # M5
        (0, 1, 0, 1, 0, 1),  # M5
        (1, 1, 3, 1, 1, 1),  # M6
        (0, 1, 2, -1, 3, 1), # M7
        (1, 1, 3, 1, 3, 1),  # M7
    ]

    def add_strassen(offset, u_base, v_base, w_base, scale=1):
        for u_idx, u_val, v_idx, v_val, w_idx, w_val in strassen:
            col = offset
            U[u_base + u_idx][col] += u_val * scale
            V[v_base + v_idx][col] += v_val * scale
            W[w_base + w_idx][col] += w_val * scale

    # Part 1: A (2x2) * B_left (2x2) -> C (2x2)
    # A cols 0,1; B rows 0,1; C cols 0,1
    # U indices: 0,1,2,3
    # V indices: 0,1, 2,3
    # W indices: 0,1, 2,3
    add_strassen(0, 0, 0, 0)

    # Part 2: A (2x2) * B_right (2x2) -> C (2x2)
    # A cols 0,1; B rows 0,1; C cols 2,3
    # U indices: 0,1,2,3
    # V indices: 4,5, 6,7
    # W indices: 4,5, 6,7
    add_strassen(7, 0, 4, 4)

    # Part 3: A_col2 (2x1) * B_row2 (1x4) -> C_col2 (2x4)
    # This is an outer product, rank 8.
    # U indices: 4,5 (A02, A12)
    # V indices: 8,9,10,11 (B20, B21, B22, B23)
    # W indices: 0..7 (C00..C03, C10..C13)
    # Term (i, k): U[i] * V[k] * W[k*n + i]
    for i in range(n):  # 0, 1
        for k in range(p):  # 0, 1, 2, 3
            col = 14 + i * p + k
            U[4 + i][col] = 1
            V[8 + k][col] = 1
            W[k * n + i][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
