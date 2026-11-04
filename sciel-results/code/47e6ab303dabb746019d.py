"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """Rank-98 decomposition via Strassen 2x2x2 tensor product, replicated for 8-wide output."""
    n, m, p = DIMENSIONS
    # Strassen 2x2x2 has rank 7
    # Tensor product for 4x4x4: rank 49
    # For 4x4x8: replicate the 4x4x4 structure for each 4x4 block of the 8-wide output
    # Actually, for 4x4x8, we use the 4x4x4 Strassen product and tile it
    # The 8-wide output means p=8, so we have 8 columns in output
    # We can decompose as: for each pair of 4x4 blocks in B (since p=8 = 2*4)
    # Actually, let me think more carefully.
    
    # For 4x4x4: rank 49 via Strassen tensor product
    # For 4x4x8: we can view B as two 4x4 blocks side by side
    # C = A * B = [A*B1 | A*B2] where B1, B2 are 4x4
    # Each A*Bk is a 4x4x4 product, rank 49 each
    # Total rank = 2 * 49 = 98
    
    # Strassen 2x2x2 factor matrices (for 2x2x2)
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # The 7 Strassen products and their contributions to C:
    # P0 = (A00+A11)(B00+B11) -> C00 += P0, C01 -= P2, C10 += P3, C11 += P0
    # P1 = (A10+A11)B00       -> C10 += P1
    # P2 = A00(B01-B11)       -> C01 += P2
    # P3 = A11(B10-B00)       -> C10 += P3
    # P4 = (A00+A01)B11       -> C11 += P4
    # P5 = (A10-A00)(B00+B10) -> C10 += P5
    # P6 = (A01-A11)(B10+B11) -> C11 += P6
    
    # So:
    # C00 = P0
    # C01 = P0 - P2
    # C10 = P1 + P3 + P5
    # C11 = P0 + P4 + P6
    
    # For 4x4x4, we take the tensor product of two 2x2x2 Strassen decompositions.
    # This gives rank 7*7 = 49.
    
    # For 4x4x8, we tile: rank 2*49 = 98.
    
    # Let me construct the 4x4x4 Strassen tensor product first, then replicate.
    
    # Strassen 2x2x2:
    # U factors (for A): shape (4, 7)
    # V factors (for B): shape (4, 7)  
    # W factors (for C): shape (4, 7)
    
    # A indices: 00->0, 01->1, 10->2, 11->3
    # B indices: 00->0, 01->1, 10->2, 11->3
    # C indices: 00->0, 01->1, 10->2, 11->3
    
    # The 7 products:
    # P0: A00+A11 times B00+B11
    # P1: A10+A11 times B00
    # P2: A00 times B01-B11
    # P3: A11 times B10-B00
    # P4: A00+A01 times B11
    # P5: A10-A00 times B00+B10
    # P6: A01-A11 times B10+B11
    
    # U_2x2 (A factors): shape (4, 7)
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1],  # A11
    ]
    
    # V_2x2 (B factors): shape (4, 7)
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 0, 0],   # B01
        [1, 0, -1, 1, 0, 1, 1],  # B10
        [0, 0, 0, 0, 1, 0, 1],   # B11
    ]
    
    # W_2x2 (C factors): shape (4, 7)
    # C00 = P0
    # C01 = P0 - P2
    # C10 = P1 + P3 + P5
    # C11 = P0 + P4 + P6
    W2 = [
        [1, 0, 0, 0, 0, 0, 0],   # C00
        [1, 0, -1, 0, 0, 0, 0],  # C01
        [0, 1, 0, 1, 0, 1, 0],   # C10
        [1, 0, 0, 0, 1, 0, 1],   # C11
    ]
    
    # Now for 4x4x4, take tensor product of two 2x2x2 Strassen.
    # The 4x4 A matrix is indexed as A[i,j] where i,j in {0,1,2,3}
    # We view it as a 2x2 block matrix of 2x2 blocks:
    # A = [[A_00, A_01], [A_10, A_11]] where each A_ij is 2x2
    # A_00 = A[0:2, 0:2], A_01 = A[0:2, 2:4], A_10 = A[2:4, 0:2], A_11 = A[2:4, 2:4]
    
    # For the tensor product, the U factor for 4x4x4 has shape (16, 49)
    # U_4[i*4+j, c1*7+c2] = U2[i, c1] * U2[j, c2]
    
    # Similarly for V and W.
    
    # But for 4x4x8, we need to handle the p=8 dimension.
    # The B matrix is 4x8, which we view as two 4x4 blocks: B[:, 0:4] and B[:, 4:8]
    # C = A * B = [A*B1 | A*B2]
    
    # For each 4x4 block of B, we use a 4x4x4 Strassen product.
    # So the total rank is 2 * 49 = 98.
    
    # Let me construct the 4x4x4 factor matrices first.
    
    # U_4: shape (16, 49)
    # Indexing: row = i*4 + j (i,j in 0..3), col = c1*7 + c2 (c1,c2 in 0..6)
    U4 = [[0]*49 for _ in range(16)]
    for i in range(2):
        for j in range(2):
            for c1 in range(7):
                for c2 in range(7):
                    val = U2[i][c1] * U2[j][c2]
                    U4[i*4+j][c1*7+c2] = val
    
    # V_4: shape (16, 49) for 4x4x4 (B is 4x4)
    # Indexing: row = j*4 + k (j,k in 0..3), col = c1*7 + c2
    V4 = [[0]*49 for _ in range(16)]
    for j in range(2):
        for k in range(2):
            for c1 in range(7):
                for c2 in range(7):
                    val = V2[j][c1] * V2[k][c2]
                    V4[j*4+k][c1*7+c2] = val
    
    # W_4: shape (16, 49) for 4x4x4 (C is 4x4)
    # Indexing: row = k*4 + i (k,i in 0..3), col = c1*7 + c2
    W4 = [[0]*49 for _ in range(16)]
    for k in range(2):
        for i in range(2):
            for c1 in range(7):
                for c2 in range(7):
                    val = W2[k][c1] * W2[i][c2]
                    W4[k*4+i][c1*7+c2] = val
    
    # Now for 4x4x8:
    # U: shape (16, 98) - same as U4, but duplicated for each B block
    # V: shape (32, 98) - B is 4x8, so rows are j*8+k (j in 0..3, k in 0..7)
    # W: shape (32, 98) - C is 4x8, so rows are k*4+i (k in 0..7, i in 0..3)
    
    rank = 98
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]
    
    # For block 0 (B columns 0-3, C columns 0-3):
    for r in range(16):
        for c in range(49):
            U[r][c] = U4[r][c]
    
    for r in range(16):
        for c in range(49):
            # V4 row r corresponds to j*4+k where j in 0..3, k in 0..3
            # For block 0, B row j, column k (k in 0..3) -> V row j*8 + k
            j = r // 4
            k = r % 4
            V[j*8 + k][c] = V4[r][c]
    
    for r in range(16):
        for c in range(49):
            # W4 row r corresponds to k*4+i where k in 0..3, i in 0..3
            # For block 0, C row i, column k (k in 0..3) -> W row k*4 + i
            # W4 is indexed as k*4+i, so r = k*4+i
            W[r][c] = W4[r][c]
    
    # For block 1 (B columns 4-7, C columns 4-7):
    for r in range(16):
        for c in range(49):
            U[r][49+c] = U4[r][c]
    
    for r in range(16):
        for c in range(49):
            j = r // 4
            k = r % 4
            # For block 1, B row j, column 4+k -> V row j*8 + (4+k)
            V[j*8 + 4 + k][49+c] = V4[r][c]
    
    for r in range(16):
        for c in range(49):
            # For block 1, C row i, column 4+k -> W row (4+k)*4 + i
            k = r // 4
            i = r % 4
            W[(4+k)*4 + i][49+c] = W4[r][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
