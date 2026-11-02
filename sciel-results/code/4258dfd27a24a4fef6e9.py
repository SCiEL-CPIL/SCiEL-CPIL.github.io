"""Optimized tensor decomposition for 2x2 * 2x13 matrix multiplication using Strassen blocks."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=13
    # Rank calculation: 6 Strassen blocks (7 rank each) + 1 Schoolbook block (4 rank)
    total_rank = 6 * 7 + 4
    rank = total_rank
    
    # U: shape (n*m, rank) = (4, rank)
    # V: shape (m*p, rank) = (26, rank)
    # W: shape (p*n, rank) = (26, rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to set a rank term
    def set_term(u_vals, v_vals, w_vals):
        nonlocal col_idx
        for i in range(len(u_vals)):
            if u_vals[i] != 0:
                U[i][col_idx] = u_vals[i]
        for idx, val in v_vals.items():
            if val != 0:
                V[idx][col_idx] = val
        for idx, val in w_vals.items():
            if val != 0:
                W[idx][col_idx] = val
        col_idx += 1

    # Process 6 Strassen blocks for columns 0..11
    for b in range(6):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Indices for V (B matrix)
        # B[0, k0] -> 0*13 + k0
        # B[0, k1] -> 0*13 + k1
        # B[1, k0] -> 1*13 + k0
        # B[1, k1] -> 1*13 + k1
        v_00 = k0
        v_01 = k1
        v_10 = 13 + k0
        v_11 = 13 + k1
        
        # Indices for W (C matrix)
        # C[i, k] -> k*2 + i
        w_00 = k0 * 2 + 0
        w_01 = k1 * 2 + 0
        w_10 = k0 * 2 + 1
        w_11 = k1 * 2 + 1

        # M1: (A00+A11)(B00+B11)
        # C00 += M1, C11 += M1
        set_term(
            [1, 0, 0, 1],
            {v_00: 1, v_11: 1},
            {w_00: 1, w_11: 1}
        )

        # M2: (A10+A11)B00
        # C10 += M2, C11 -= M2
        set_term(
            [0, 0, 1, 1],
            {v_00: 1},
            {w_10: 1, w_11: -1}
        )

        # M3: A00(B10-B11)
        # C01 += M3, C11 += M3
        set_term(
            [1, 0, 0, 0],
            {v_10: 1, v_11: -1},
            {w_01: 1, w_11: 1}
        )

        # M4: A11(B01+B11)
        # C00 += M4, C10 += M4
        set_term(
            [0, 0, 0, 1],
            {v_01: 1, v_11: 1},
            {w_00: 1, w_10: 1}
        )

        # M5: (A00+A01)B11
        # C00 -= M5, C01 += M5
        set_term(
            [1, 1, 0, 0],
            {v_11: 1},
            {w_00: -1, w_01: 1}
        )

        # M6: (A10-A00)(B00+B01)
        # C11 += M6
        set_term(
            [-1, 0, 1, 0],
            {v_00: 1, v_01: 1},
            {w_11: 1}
        )

        # M7: (A01-A11)(B10+B11)
        # C00 += M7
        set_term(
            [0, 1, 0, -1],
            {v_10: 1, v_11: 1},
            {w_00: 1}
        )

    # Process 1 Schoolbook block for column 12
    k = 12
    v_0 = k
    v_1 = 13 + k
    w_0 = k * 2 + 0
    w_1 = k * 2 + 1

    # Term 1: A00 * B0k -> C0k
    set_term(
        [1, 0, 0, 0],
        {v_0: 1},
        {w_0: 1}
    )
    
    # Term 2: A10 * B1k -> C0k
    set_term(
        [0, 0, 1, 0],
        {v_1: 1},
        {w_0: 1}
    )
    
    # Term 3: A01 * B0k -> C1k
    set_term(
        [0, 1, 0, 0],
        {v_0: 1},
        {w_1: 1}
    )
    
    # Term 4: A11 * B1k -> C1k
    set_term(
        [0, 0, 0, 1],
        {v_1: 1},
        {w_1: 1}
    )

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
