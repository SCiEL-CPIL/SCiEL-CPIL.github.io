import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _strassen_4x4_decomposition():
    """
    Construct the rank-49 Strassen decomposition for 4x4 matrix multiplication.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Outer Strassen: 7 block products
    # A blocks: 00, 01, 10, 11 (row_block, col_block)
    # Each block is 2x2.
    # Outer product i combines A blocks and B blocks.
    
    # A_block_combos[i] is a list of (block_row, block_col, coeff)
    # B_block_combos[i] is a list of (block_row, block_col, coeff)
    
    A_block_combos = [
        [(0, 0, 1), (1, 1, 1)],       # M1: A00 + A11
        [(1, 0, 1), (1, 1, 1)],       # M2: A10 + A11
        [(0, 0, 1)],                  # M3: A00
        [(1, 1, 1)],                  # M4: A11
        [(0, 0, 1), (0, 1, 1)],       # M5: A00 + A01
        [(1, 0, 1), (0, 0, -1)],      # M6: A10 - A00
        [(0, 1, 1), (1, 1, -1)],      # M7: A01 - A11
    ]
    
    B_block_combos = [
        [(0, 0, 1), (1, 1, 1)],       # M1: B00 + B11
        [(0, 0, 1)],                  # M2: B00
        [(0, 1, 1), (1, 1, -1)],      # M3: B01 - B11
        [(1, 0, 1), (0, 0, -1)],      # M4: B10 - B00
        [(1, 1, 1)],                  # M5: B11
        [(0, 0, 1), (0, 1, 1)],       # M6: B00 + B01
        [(1, 0, 1), (1, 1, 1)],       # M7: B10 + B11
    ]
    
    # Inner Strassen: 7 scalar products within a 2x2 block
    # Local indices (lr, lc) for 2x2 block.
    
    A_local_combos = [
        [(0, 0, 1), (1, 1, 1)],       # P1: a + d
        [(1, 0, 1), (1, 1, 1)],       # P2: c + d
        [(0, 0, 1)],                  # P3: a
        [(1, 1, 1)],                  # P4: d
        [(0, 0, 1), (0, 1, 1)],       # P5: a + b
        [(1, 0, 1), (0, 0, -1)],      # P6: c - a
        [(0, 1, 1), (1, 1, -1)],      # P7: b - d
    ]
    
    B_local_combos = [
        [(0, 0, 1), (1, 1, 1)],       # P1: e + h
        [(0, 0, 1)],                  # P2: e
        [(0, 1, 1), (1, 1, -1)],      # P3: f - h
        [(1, 0, 1), (0, 0, -1)],      # P4: g - e
        [(1, 1, 1)],                  # P5: h
        [(0, 0, 1), (0, 1, 1)],       # P6: e + f
        [(1, 0, 1), (1, 1, 1)],       # P7: g + h
    ]
    
    # C contribution weights
    # Inner: weight of product j for local C entry (lr, lc)
    # C_local[0,0] = P1 + P4 - P5 + P7
    # C_local[0,1] = P3 + P5
    # C_local[1,0] = P2 + P4
    # C_local[1,1] = P1 - P2 + P3 + P6
    
    C_local_weights = [
        # j=0 (P1)
        [(0, 0, 1), (1, 1, 1)],
        # j=1 (P2)
        [(1, 0, 1), (1, 1, -1)],
        # j=2 (P3)
        [(0, 1, 1), (1, 1, 1)],
        # j=3 (P4)
        [(0, 0, 1), (1, 0, 1)],
        # j=4 (P5)
        [(0, 0, -1), (0, 1, 1)],
        # j=5 (P6)
        [(1, 1, 1)],
        # j=6 (P7)
        [(0, 0, 1)],
    ]
    
    # Outer: weight of block product i for global C block (br, bc)
    # C_block[0,0] = M1 + M4 - M5 + M7
    # C_block[0,1] = M3 + M5
    # C_block[1,0] = M2 + M4
    # C_block[1,1] = M1 - M2 + M3 + M6
    
    C_block_weights = [
        # i=0 (M1)
        [(0, 0, 1), (1, 1, 1)],
        # i=1 (M2)
        [(1, 0, 1), (1, 1, -1)],
        # i=2 (M3)
        [(0, 1, 1), (1, 1, 1)],
        # i=3 (M4)
        [(0, 0, 1), (1, 0, 1)],
        # i=4 (M5)
        [(0, 0, -1), (0, 1, 1)],
        # i=5 (M6)
        [(1, 1, 1)],
        # i=6 (M7)
        [(0, 0, 1)],
    ]
    
    # Construct U, V, W
    for i in range(7):
        for j in range(7):
            r_idx = i * 7 + j
            
            # U contributions
            # U is indexed by (row, col) of A.
            # Global row = 2 * block_row + local_row
            # Global col = 2 * block_col + local_col
            for (abr, abc, a_coeff) in A_block_combos[i]:
                for (lr, lc, l_coeff) in A_local_combos[j]:
                    global_row = 2 * abr + lr
                    global_col = 2 * abc + lc
                    U[global_row * m + global_col][r_idx] += a_coeff * l_coeff
            
            # V contributions
            # V is indexed by (row, col) of B.
            # Global row = 2 * block_row + local_row
            # Global col = 2 * block_col + local_col
            for (bbr, bbc, b_coeff) in B_block_combos[i]:
                for (lr, lc, l_coeff) in B_local_combos[j]:
                    global_row = 2 * bbr + lr
                    global_col = 2 * bbc + lc
                    V[global_row * p + global_col][r_idx] += b_coeff * l_coeff
            
            # W contributions
            # W is indexed by W[k*n + i] where k is col, i is row of C.
            # Global row R = 2 * c_block_row + c_local_row
            # Global col C = 2 * c_block_col + c_local_col
            # W index = C * n + R
            for (cbr, cbc, c_coeff) in C_block_weights[i]:
                for (lr, lc, l_coeff) in C_local_weights[j]:
                    global_row = 2 * cbr + lr
                    global_col = 2 * cbc + lc
                    w_idx = global_col * n + global_row
                    W[w_idx][r_idx] += c_coeff * l_coeff
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Fallback schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS

    # Try the recursive Strassen rank-49 decomposition
    result = _strassen_4x4_decomposition()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])

    # Verify correctness
    if _verify_decomposition(U, V, W, n, m, p, rank):
        return result

    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)

# EVOLVE-BLOCK-END
