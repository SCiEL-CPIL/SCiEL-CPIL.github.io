import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    # n=2, m=3, p=7
    # U shape (6, R)
    # V shape (21, R)
    # W shape (14, R)
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block for 2x2x2
    # a_indices: indices in U for A00, A01, A10, A11
    # b_indices: indices in V for B00, B01, B10, B11
    # w_indices: indices in W for C00, C01, C10, C11
    def add_strassen_block(a_idx, b_idx, w_idx):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        U[a_idx[0]][col] += 1
        U[a_idx[1]][col] += 1
        V[b_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        W[w_idx[0]][col] += 1
        W[w_idx[1]][col] += 1
        W[w_idx[2]][col] += 1
        W[w_idx[3]][col] += 1
        col += 1
        
        # M2 = (A10 + A11) B00
        U[a_idx[2]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        W[w_idx[2]][col] += 1
        W[w_idx[3]][col] -= 1
        col += 1
        
        # M3 = A00 (B01 - B11)
        U[a_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        V[b_idx[3]][col] -= 1
        W[w_idx[1]][col] += 1
        W[w_idx[3]][col] += 1
        col += 1
        
        # M4 = A11 (B10 - B00)
        U[a_idx[3]][col] += 1
        V[b_idx[2]][col] += 1
        V[b_idx[0]][col] -= 1
        W[w_idx[0]][col] += 1
        W[w_idx[2]][col] += 1
        col += 1
        
        # M5 = (A00 + A01) B11
        U[a_idx[0]][col] += 1
        U[a_idx[1]][col] += 1
        V[b_idx[3]][col] += 1
        W[w_idx[0]][col] -= 1
        W[w_idx[1]][col] += 1
        col += 1
        
        # M6 = (A01 - A10)(B00 + B10)
        U[a_idx[1]][col] += 1
        U[a_idx[2]][col] -= 1
        V[b_idx[0]][col] += 1
        V[b_idx[2]][col] += 1
        W[w_idx[3]][col] += 1
        col += 1
        
        # M7 = (A10 - A00)(B01 + B11)
        U[a_idx[2]][col] += 1
        U[a_idx[0]][col] -= 1
        V[b_idx[1]][col] += 1
        V[b_idx[3]][col] += 1
        W[w_idx[0]][col] += 1
        W[w_idx[2]][col] -= 1
        col += 1

    # Helper to add a Schoolbook block for 2x2x1
    def add_schoolbook_2x2x1(a_idx, b_idx, w_idx):
        nonlocal col
        # C00 = A00 B00 + A01 B10
        # C10 = A10 B00 + A11 B10
        # b_idx: B00, B10
        # w_idx: C00, C10
        
        # A00 B00
        U[a_idx[0]][col] += 1
        V[b_idx[0]][col] += 1
        W[w_idx[0]][col] += 1
        col += 1
        
        # A01 B10
        U[a_idx[1]][col] += 1
        V[b_idx[1]][col] += 1
        W[w_idx[0]][col] += 1
        col += 1
        
        # A10 B00
        U[a_idx[2]][col] += 1
        V[b_idx[0]][col] += 1
        W[w_idx[1]][col] += 1
        col += 1
        
        # A11 B10
        U[a_idx[3]][col] += 1
        V[b_idx[1]][col] += 1
        W[w_idx[1]][col] += 1
        col += 1

    # Helper to add Schoolbook for 2x1x7
    # a_idx: A0, A1 (indices in U)
    # b_idx: B0..B6 (indices in V)
    # w_idx: C00..C16 (indices in W)
    def add_schoolbook_2x1x7(a_idx, b_idx, w_idx):
        nonlocal col
        # For each k in 0..6
        # C0k += A0 Bk
        # C1k += A1 Bk
        for k in range(7):
            # A0 Bk
            U[a_idx[0]][col] += 1
            V[b_idx[k]][col] += 1
            W[w_idx[k]][col] += 1
            col += 1
            
            # A1 Bk
            U[a_idx[1]][col] += 1
            V[b_idx[k]][col] += 1
            W[w_idx[k+7]][col] += 1
            col += 1

    # Define indices for A (2x3)
    # A[i, j] is at U[i*3 + j]
    A_00 = 0
    A_01 = 1
    A_02 = 2
    A_10 = 3
    A_11 = 4
    A_12 = 5
    
    # Define indices for B (3x7)
    # B[j, k] is at V[j*7 + k]
    B_00 = 0
    B_01 = 1
    B_02 = 2
    B_03 = 3
    B_04 = 4
    B_05 = 5
    B_06 = 6
    B_10 = 7
    B_11 = 8
    B_12 = 9
    B_13 = 10
    B_14 = 11
    B_15 = 12
    B_16 = 13
    B_20 = 14
    B_21 = 15
    B_22 = 16
    B_23 = 17
    B_24 = 18
    B_25 = 19
    B_26 = 20
    
    # Define indices for W (7x2)
    # W[k, i] is at W[k*2 + i]
    # k is col (0..6), i is row (0..1)
    # C[i, k]
    C_00 = 0
    C_10 = 1
    C_01 = 2
    C_11 = 3
    C_02 = 4
    C_12 = 5
    C_03 = 6
    C_13 = 7
    C_04 = 8
    C_14 = 9
    C_05 = 10
    C_15 = 11
    C_06 = 12
    C_16 = 13

    # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
    # A indices: 0, 1, 3, 4
    # B indices: 0, 1, 7, 8
    # W indices: 0, 2, 1, 3
    add_strassen_block([A_00, A_01, A_10, A_11], [B_00, B_01, B_10, B_11], [C_00, C_01, C_10, C_11])
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
    # A indices: 0, 1, 3, 4
    # B indices: 2, 3, 9, 10
    # W indices: 4, 6, 5, 7
    add_strassen_block([A_00, A_01, A_10, A_11], [B_02, B_03, B_12, B_13], [C_02, C_03, C_12, C_13])
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Block 3: Strassen for A[0:2, 0:2] * B[0:2, 4:6]
    # A indices: 0, 1, 3, 4
    # B indices: 4, 5, 11, 12
    # W indices: 8, 10, 9, 11
    add_strassen_block([A_00, A_01, A_10, A_11], [B_04, B_05, B_14, B_15], [C_04, C_05, C_14, C_15])
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Block 4: Schoolbook for A[0:2, 0:2] * B[0:2, 6:7]
    # A indices: 0, 1, 3, 4
    # B indices: 6, 13
    # W indices: 12, 13
    add_schoolbook_2x2x1([A_00, A_01, A_10, A_11], [B_06, B_16], [C_06, C_16])
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    # Block 5: Schoolbook for A[0:2, 2:3] * B[2:3, 0:7]
    # A indices: 2, 5
    # B indices: 14, 15, 16, 17, 18, 19, 20
    # W indices: 0, 2, 4, 6, 8, 10, 12 (for row 0)
    #           1, 3, 5, 7, 9, 11, 13 (for row 1)
    w_row0 = [C_00, C_01, C_02, C_03, C_04, C_05, C_06]
    w_row1 = [C_10, C_11, C_12, C_13, C_14, C_15, C_16]
    b_cols = [B_20, B_21, B_22, B_23, B_24, B_25, B_26]
    add_schoolbook_2x1x7([A_02, A_12], b_cols, w_row0 + w_row1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
