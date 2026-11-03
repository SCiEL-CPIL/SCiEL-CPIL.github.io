import numpy as np

DIMENSIONS = [4, 4, 8]

def get_444_factors():
    """
    Returns U, V, W factors for 4x4x4 matrix multiplication with rank 49.
    U, V, W are lists of lists with shape (16, 49).
    """
    # Strassen 2x2x2 factors
    # U_inner: 4x7, V_inner: 4x7, W_inner: 4x7
    # Indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    U_inner = [
        [1, 0, 1, 0, 1, -1, 0], # M0: A00+A11, M1: A10+A11, M2: A00, M3: A11, M4: A00+A01, M5: A10-A00, M6: A01-A11
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    
    V_inner = [
        [1, 1, 0, -1, 0, 1, 0], # N0: B00+B11, N1: B00, N2: B01-B11, N3: B10-B00, N4: B11, N5: B00+B10, N6: B10+B11
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    
    W_inner = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, 0, -1, 0, 0, 1, 1]   # C11
    ]
    
    # Outer combinations for 4x4x4
    # A blocks:
    # A00: indices 0,1,4,5
    # A01: indices 2,3,6,7
    # A10: indices 8,9,12,13
    # A11: indices 10,11,14,15
    
    # M0 = A00 + A11
    # M1 = A10 + A11
    # M2 = A00
    # M3 = A11
    # M4 = A00 + A01
    # M5 = A10 - A00
    # M6 = A01 - A11
    
    # We need to map each entry of M_k to the global A indices.
    # M_k is 2x2.
    # M_k[0,0] corresponds to A_block[0,0]
    # M_k[0,1] corresponds to A_block[0,1]
    # M_k[1,0] corresponds to A_block[1,0]
    # M_k[1,1] corresponds to A_block[1,1]
    
    # Block indices in global A:
    # A00: [[0,1],[4,5]]
    # A01: [[2,3],[6,7]]
    # A10: [[8,9],[12,13]]
    # A11: [[10,11],[14,15]]
    
    # Define the linear combinations for M_k entries
    # m_k[r, c] = sum(coeff * A[global_idx])
    
    # We will build U_444, V_444, W_444 of shape 16x49
    U_444 = [[0]*49 for _ in range(16)]
    V_444 = [[0]*49 for _ in range(16)]
    W_444 = [[0]*49 for _ in range(16)]
    
    # Helper to add a block to a matrix of coefficients
    # block_coeffs: 2x2 matrix of coefficients for the block
    # block_indices: 2x2 matrix of global indices
    # target: 16x49 matrix
    # col_offset: offset in the 49 columns
    
    def add_block_contribution(target, block_indices, block_coeffs, col_offset):
        for r in range(2):
            for c in range(2):
                coeff = block_coeffs[r][c]
                if coeff != 0:
                    idx = block_indices[r][c]
                    # The entry M_k[r,c] contributes to the l-th inner factor
                    # with weight U_inner[r*2+c, l]
                    for l in range(7):
                        w = U_inner[r*2+c][l]
                        if w != 0:
                            target[idx][col_offset + l] += coeff * w

    # Define block indices
    A00_idx = [[0, 1], [4, 5]]
    A01_idx = [[2, 3], [6, 7]]
    A10_idx = [[8, 9], [12, 13]]
    A11_idx = [[10, 11], [14, 15]]
    
    # Define M_k block coefficients
    # M0 = A00 + A11
    M0_coeffs = [
        [[1, 0], [0, 1]], # from A00
        [[1, 0], [0, 1]]  # from A11
    ]
    # This structure is awkward. Let's just define the 2x2 matrix of coefficients for each block.
    
    # Let's restructure.
    # For each k in 0..6:
    #   M_k is sum of blocks.
    #   For each block B in [A00, A01, A10, A11]:
    #     coeff_matrix is 2x2.
    #     Add contribution.
    
    # M0: A00 + A11
    # M1: A10 + A11
    # M2: A00
    # M3: A11
    # M4: A00 + A01
    # M5: A10 - A00
    # M6: A01 - A11
    
    I = [[1, 0], [0, 1]]
    Z = [[0, 0], [0, 0]]
    NegI = [[-1, 0], [0, -1]]
    
    # M0
    add_block_contribution(U_444, A00_idx, I, 0)
    add_block_contribution(U_444, A11_idx, I, 0)
    
    # M1
    add_block_contribution(U_444, A10_idx, I, 7)
    add_block_contribution(U_444, A11_idx, I, 7)
    
    # M2
    add_block_contribution(U_444, A00_idx, I, 14)
    
    # M3
    add_block_contribution(U_444, A11_idx, I, 21)
    
    # M4
    add_block_contribution(U_444, A00_idx, I, 28)
    add_block_contribution(U_444, A01_idx, I, 28)
    
    # M5
    add_block_contribution(U_444, A10_idx, I, 35)
    add_block_contribution(U_444, A00_idx, NegI, 35)
    
    # M6
    add_block_contribution(U_444, A01_idx, I, 42)
    add_block_contribution(U_444, A11_idx, NegI, 42)
    
    # Now V_444
    # N0 = B00 + B11
    # N1 = B00
    # N2 = B01 - B11
    # N3 = B10 - B00
    # N4 = B11
    # N5 = B00 + B10
    # N6 = B10 + B11
    
    # B blocks indices are same as A (0-15)
    
    # N0
    add_block_contribution(V_444, A00_idx, I, 0) # Using A00_idx as B00_idx
    add_block_contribution(V_444, A11_idx, I, 0) # Using A11_idx as B11_idx
    
    # N1
    add_block_contribution(V_444, A00_idx, I, 7)
    
    # N2
    add_block_contribution(V_444, A01_idx, I, 14)
    add_block_contribution(V_444, A11_idx, NegI, 14)
    
    # N3
    add_block_contribution(V_444, A10_idx, I, 21)
    add_block_contribution(V_444, A00_idx, NegI, 21)
    
    # N4
    add_block_contribution(V_444, A11_idx, I, 28)
    
    # N5
    add_block_contribution(V_444, A00_idx, I, 35)
    add_block_contribution(V_444, A10_idx, I, 35)
    
    # N6
    add_block_contribution(V_444, A10_idx, I, 42)
    add_block_contribution(V_444, A11_idx, I, 42)
    
    # Now W_444
    # C blocks indices are same as A (0-15)
    # C00, C01, C10, C11
    # P0 = M0 N0
    # P1 = M1 N1
    # P2 = M2 N2
    # P3 = M3 N3
    # P4 = M4 N4
    # P5 = M5 N5
    # P6 = M6 N6
    
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P2 + P5 + P6
    
    # We need to add contributions from each P_k to the corresponding C blocks.
    # P_k is a 2x2 matrix.
    # The entries of P_k are linear combinations of the 7 scalar products x_{k,l}.
    # W_inner[r*2+c, l] gives the coefficient of x_{k,l} in P_k[r,c].
    
    # For each k, and for each C block that includes P_k:
    #   For each entry (r,c) of the C block:
    #     The contribution from P_k is W_inner[r*2+c, l] * x_{k,l}
    #     So we add W_inner[r*2+c, l] to W_444[global_idx, k*7+l]
    
    # C00 is at indices 0,1,4,5.
    # C01 is at indices 2,3,6,7.
    # C10 is at indices 8,9,12,13.
    # C11 is at indices 10,11,14,15.
    
    def add_P_contribution(target, block_indices, W_inner_row, col_offset):
        # W_inner_row is a list of 7 coefficients for the 7 scalar products
        for r in range(2):
            for c in range(2):
                idx = block_indices[r][c]
                # The coefficient for x_{k,l} is W_inner[r*2+c, l]
                # We need to add this to target[idx, col_offset + l]
                for l in range(7):
                    w = W_inner[r*2+c][l]
                    if w != 0:
                        target[idx][col_offset + l] += w

    # C00 = P0 + P3 - P4 + P6
    add_P_contribution(W_444, A00_idx, W_inner[0], 0) # P0
    add_P_contribution(W_444, A00_idx, W_inner[0], 21) # P3 (col offset 3*7=21)
    # Wait, W_inner[0] is for C00 of the 2x2 product.
    # The 2x2 product P_k has entries P_k[0,0], P_k[0,1], P_k[1,0], P_k[1,1].
    # These correspond to the entries of the C block.
    # So for C00, we use the rows of W_inner corresponding to C00, C01, C10, C11 of the 2x2 product?
    # No. P_k is a 2x2 matrix.
    # P_k[0,0] contributes to C00[0,0].
    # P_k[0,1] contributes to C00[0,1].
    # P_k[1,0] contributes to C00[1,0].
    # P_k[1,1] contributes to C00[1,1].
    
    # So for each P_k, we add its 4 entries to the corresponding C block.
    
    # Let's refine add_P_contribution.
    # It should take the 4 rows of W_inner (one for each entry of the 2x2 product).
    # W_inner is 4x7. Rows 0,1,2,3 correspond to C00, C01, C10, C11 of the 2x2 product.
    
    def add_P_block(target, block_indices, W_inner_full, col_offset):
        # W_inner_full is 4x7
        for r in range(2):
            for c in range(2):
                idx = block_indices[r][c]
                # The entry (r,c) of the product corresponds to W_inner_full[r*2+c]
                w_row = W_inner_full[r*2+c]
                for l in range(7):
                    w = w_row[l]
                    if w != 0:
                        target[idx][col_offset + l] += w

    # C00 = P0 + P3 - P4 + P6
    add_P_block(W_444, A00_idx, W_inner, 0) # P0
    add_P_block(W_444, A00_idx, W_inner, 21) # P3
    # P4 has coefficient -1
    for r in range(2):
        for c in range(2):
            idx = A00_idx[r][c]
            w_row = W_inner[r*2+c]
            for l in range(7):
                if w_row[l] != 0:
                    W_444[idx][28 + l] -= w_row[l]
    # P6 has coefficient 1
    add_P_block(W_444, A00_idx, W_inner, 42) # P6
    
    # C01 = P2 + P4
    add_P_block(W_444, A01_idx, W_inner, 14) # P2
    add_P_block(W_444, A01_idx, W_inner, 28) # P4
    
    # C10 = P1 + P3
    add_P_block(W_444, A10_idx, W_inner, 7) # P1
    add_P_block(W_444, A10_idx, W_inner, 21) # P3
    
    # C11 = P0 - P2 + P5 + P6
    add_P_block(W_444, A11_idx, W_inner, 0) # P0
    # P2 has coefficient -1
    for r in range(2):
        for c in range(2):
            idx = A11_idx[r][c]
            w_row = W_inner[r*2+c]
            for l in range(7):
                if w_row[l] != 0:
                    W_444[idx][14 + l] -= w_row[l]
    add_P_block(W_444, A11_idx, W_inner, 35) # P5
    add_P_block(W_444, A11_idx, W_inner, 42) # P6
    
    return U_444, V_444, W_444

def construct():
    n, m, p = DIMENSIONS
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_444, V_444, W_444 = get_444_factors()
    
    # First 49 columns
    for i in range(16):
        for col in range(49):
            U[i][col] = U_444[i][col]
            
    # V for first block: rows 0-15 (k=0..3)
    for j in range(4):
        for k in range(4):
            for col in range(49):
                V[j * 8 + k][col] = V_444[j * 4 + k][col]
                
    # W for first block: rows 0-15 (k=0..3)
    for k in range(4):
        for i in range(4):
            for col in range(49):
                W[k * 4 + i][col] = W_444[k * 4 + i][col]
                
    # Last 49 columns
    for i in range(16):
        for col in range(49):
            U[i][col + 49] = U_444[i][col]
            
    # V for second block: rows 16-31 (k=4..7)
    for j in range(4):
        for k in range(4, 8):
            for col in range(49):
                V[j * 8 + k][col + 49] = V_444[j * 4 + (k - 4)][col]
                
    # W for second block: rows 16-31 (k=4..7)
    for k in range(4, 8):
        for i in range(4):
            for col in range(49):
                W[k * 4 + i][col + 49] = W_444[(k - 4) * 4 + i][col]
                
    return {"U": U, "V": V, "W": W}
