# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    # We construct a rank-16 decomposition by splitting p=4 into two groups of 2.
    # Each group uses a rank-8 decomposition for 2x3x2 matrix multiplication.
    # The rank-8 decomposition for 2x3x2 consists of:
    # 1. A Strassen rank-7 decomposition for the 2x2x2 part (indices j=0,1).
    # 2. A rank-1 decomposition for the remaining j=2 part.
    
    # Strassen 2x2x2 decomposition (Rank 7)
    # A (2x2), B (2x2), C (2x2)
    # C = A B
    # U_S (4x7), V_S (4x7), W_S (4x7)
    # Indices for 2x2x2:
    # u = i*2 + j, v = j*2 + k, w = k*2 + i
    # Strassen terms:
    # m1 = (a00 + a11)(b00 + b11)
    # m2 = (a10 + a11)b00
    # m3 = a00(b01 - b11)
    # m4 = a11(b10 - b00)
    # m5 = (a00 + a01)b11
    # m6 = (a10 - a00)(b00 + b01)
    # m7 = (a01 - a11)(b10 + b11)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    U_S = [
        [1, 0, 1, 0, 1, 1, 0], # a00
        [0, 0, 0, 1, 0, 0, 1], # a01
        [0, 1, 0, 0, 0, 1, 0], # a10
        [1, 1, 0, 1, 1, 0, 1]  # a11
    ]
    V_S = [
        [1, 1, 0, -1, 0, 1, 0], # b00
        [0, 0, 1, 0, 1, 1, 0], # b01
        [1, 0, 0, 1, 0, 0, 1], # b10
        [1, 0, -1, 0, 1, 0, 1] # b11
    ]
    W_S = [
        [1, 0, 0, 1, -1, 0, 1], # c00
        [0, 0, 1, 0, 1, 0, 0], # c01
        [0, 1, 0, 1, 0, 0, 0], # c10
        [1, -1, 1, 0, 0, 1, 1] # c11
    ]

    # Rank-1 part for j=2 in 2x3x2
    # A_col2 = [a02, a12], B_row2 = [b20, b21]
    # U_R1 (2x1), V_R1 (2x1), W_R1 (2x1) -- wait, dimensions
    # For 2x3x2:
    # U (6x1), V (6x1), W (4x1)
    # U_R1: non-zero at i=0, j=2 (idx 2) and i=1, j=2 (idx 5)
    # V_R1: non-zero at j=2, k=0 (idx 4) and j=2, k=1 (idx 5)
    # W_R1: non-zero at k=0, i=0 (idx 0), k=0, i=1 (idx 1), k=1, i=0 (idx 2), k=1, i=1 (idx 3)
    # Actually, the rank-1 term is:
    # U_{i, 0} = delta_{j, 2} (from A side? No, U is A side)
    # U_{i*3+j, 0} = 1 if j=2.
    # V_{j*2+k, 0} = 1 if j=2.
    # W_{k*2+i, 0} = 1.
    
    U_R1_2x3x2 = [0, 0, 1, 0, 0, 1] # 6x1
    V_R1_2x3x2 = [0, 0, 0, 0, 1, 1] # 6x1 (j=2, k=0 is idx 4; j=2, k=1 is idx 5)
    W_R1_2x3x2 = [1, 1, 1, 1]       # 4x1

    # Construct full 2x3x4 decomposition
    rank = 16
    U = [[0] * rank for _ in range(n * m)] # 6 x 16
    V = [[0] * rank for _ in range(m * p)] # 12 x 16
    W = [[0] * rank for _ in range(p * n)] # 8 x 16

    # Group 1: k in {0, 1}
    # U part: same for both groups
    for c in range(7):
        for i in range(2):
            for j in range(2):
                # Map 2x2 indices to 2x3 indices
                # u_idx = i*3 + j
                U[i * 3 + j][c] = U_S[i * 2 + j][c]
        for j in range(2):
            for k in range(2):
                # Map 2x2 indices to 2x3x2 V indices
                # v_idx = j*2 + k
                # In full 2x3x4, V index is j*4 + k
                # For group 1, k is 0,1. So v_full = j*4 + k
                V[j * 4 + k][c] = V_S[j * 2 + k][c]
        for k in range(2):
            for i in range(2):
                # Map 2x2 indices to 2x2 W indices
                # w_idx = k*2 + i
                # In full 2x3x4, W index is k*2 + i
                W[k * 2 + i][c] = W_S[k * 2 + i][c]

    # Rank 1 part for group 1 (j=2, k in {0,1})
    c = 7
    for i in range(2):
        U[i * 3 + 2][c] = U_R1_2x3x2[i * 3 + 2]
    for j in range(3):
        for k in range(2):
            if j == 2:
                V[j * 4 + k][c] = V_R1_2x3x2[j * 2 + k] # j=2, k=0->4, k=1->5
    for k in range(2):
        for i in range(2):
            W[k * 2 + i][c] = W_R1_2x3x2[k * 2 + i]

    # Group 2: k in {2, 3}
    # U part: same
    for c in range(7):
        c_full = 8 + c
        for i in range(2):
            for j in range(2):
                U[i * 3 + j][c_full] = U_S[i * 2 + j][c]
        for j in range(2):
            for k in range(2):
                # k_full = 2 + k
                V[j * 4 + (2 + k)][c_full] = V_S[j * 2 + k][c]
        for k in range(2):
            for i in range(2):
                # k_full = 2 + k
                W[(2 + k) * 2 + i][c_full] = W_S[k * 2 + i][c]

    # Rank 1 part for group 2 (j=2, k in {2,3})
    c = 15
    for i in range(2):
        U[i * 3 + 2][c] = U_R1_2x3x2[i * 3 + 2]
    for j in range(3):
        for k in range(2):
            if j == 2:
                # k_full = 2 + k
                V[j * 4 + (2 + k)][c] = V_R1_2x3x2[j * 2 + k]
    for k in range(2):
        for i in range(2):
            # k_full = 2 + k
            W[(2 + k) * 2 + i][c] = W_R1_2x3x2[k * 2 + i]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
