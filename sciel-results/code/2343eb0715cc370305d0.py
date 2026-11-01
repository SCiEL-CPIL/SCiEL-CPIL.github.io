import numpy as np

def construct():
    n, m, p = 2, 3, 7
    rank = 39
    
    # Initialize matrices
    # U: (n*m, rank) = (6, 39)
    # V: (m*p, rank) = (21, 39)
    # W: (p*n, rank) = (14, 39)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 B1 (2x1x7)
    # A1 is 2x1, B1 is 1x7.
    # C1 is 2x7.
    # C1[i,k] = A1[i,0] * B1[0,k]
    for i in range(n):
        for k in range(p):
            # U index: i (since A1 is first 1 col of A, row i)
            U[i][col] = 1
            # V index: k (since B1 is first row of B, col k)
            V[k][col] = 1
            # W index: k*n + i
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A2 B2 (2x2x7)
    # A2 is 2x2 (cols 1,2 of A), B2 is 2x7 (rows 1,2 of B).
    # Split B2 into 3 blocks of width 2 and 1 block of width 1.
    
    # Helper to add a Strassen block (2x2x2)
    def add_strassen_block(offset_U, offset_V, offset_W, start_col):
        # Local indices for A2: 0,1,2,3 -> a00, a01, a10, a11
        # Global U indices: offset_U + [0, 1, 2, 3]
        # Local indices for B_block: 0,1,2,3 -> b00, b01, b10, b11
        # Global V indices: offset_V + [0, 1, 2, 3]
        # Local indices for C_block: 0,1,2,3 -> c00, c01, c10, c11
        # Global W indices: offset_W + [0, 1, 2, 3]
        
        # Strassen terms
        # U_local, V_local, W_local
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # m1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]), # m2
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, -1]), # m3
            ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 1, 0]), # m4
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]), # m5
            ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0]), # m6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0]), # m7
        ]
        
        c = start_col
        for u_loc, v_loc, w_loc in terms:
            for l in range(4):
                U[offset_U + l][c] += u_loc[l]
                V[offset_V + l][c] += v_loc[l]
                W[offset_W + l][c] += w_loc[l]
            c += 1
        return c

    # A2 global rows: 2, 3, 4, 5
    offset_U_A2 = 2
    
    # Block 2.1: k=0,1
    # B rows: 7,8 (j=0), 14,15 (j=1)
    # C rows: 0,2 (k=0), 1,3 (k=1) -> indices 0,2,1,3
    offset_V_B2_1 = 7
    offset_W_C2_1 = 0
    # Note: The local order for V is b00, b01, b10, b11.
    # Global V indices: 7, 8, 14, 15.
    # We need to map local index l to global index.
    # l=0 -> 7, l=1 -> 8, l=2 -> 14, l=3 -> 15.
    # This is NOT contiguous. My add_strassen_block assumes contiguous offsets.
    # I need to modify the helper or pass explicit index lists.
    
    # Let's rewrite the helper to take index lists.
    
    def add_block(U_idx, V_idx, W_idx, start_col):
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, -1]),
            ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 1, 0]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0]),
        ]
        c = start_col
        for u_loc, v_loc, w_loc in terms:
            for l in range(4):
                U[U_idx[l]][c] += u_loc[l]
                V[V_idx[l]][c] += v_loc[l]
                W[W_idx[l]][c] += w_loc[l]
            c += 1
        return c

    # A2 indices: [2, 3, 4, 5]
    U_idx_A2 = [2, 3, 4, 5]
    
    # Block 2.1 (k=0,1)
    V_idx_1 = [7, 8, 14, 15]
    W_idx_1 = [0, 2, 1, 3]
    col = add_block(U_idx_A2, V_idx_1, W_idx_1, col)
    
    # Block 2.2 (k=2,3)
    V_idx_2 = [9, 10, 16, 17]
    W_idx_2 = [4, 6, 5, 7]
    col = add_block(U_idx_A2, V_idx_2, W_idx_2, col)
    
    # Block 2.3 (k=4,5)
    V_idx_3 = [11, 12, 18, 19]
    W_idx_3 = [8, 10, 9, 11]
    col = add_block(U_idx_A2, V_idx_3, W_idx_3, col)
    
    # Block 2.4 (k=6) - 2x2x1
    # a00 b06 -> c06
    U[2][col] += 1
    V[13][col] += 1
    W[12][col] += 1
    col += 1
    # a01 b16 -> c06
    U[3][col] += 1
    V[20][col] += 1
    W[12][col] += 1
    col += 1
    # a10 b06 -> c16
    U[4][col] += 1
    V[13][col] += 1
    W[13][col] += 1
    col += 1
    # a11 b16 -> c16
    U[5][col] += 1
    V[20][col] += 1
    W[13][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
