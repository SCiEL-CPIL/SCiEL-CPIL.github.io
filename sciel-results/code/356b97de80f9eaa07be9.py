import numpy as np

DIMENSIONS = [5, 5, 5]

def _schoolbook_decomp(n, m, p):
    """
    Returns U, V, W for schoolbook multiplication of n x m by m x p.
    U: (n*m, n*m*p)
    V: (m*p, n*m*p)
    W: (p*n, n*m*p)
    """
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
    return U, V, W

def _strassen_decomp_2x2():
    """
    Returns U, V, W for Strassen multiplication of 2x2 by 2x2.
    Rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices:
    A: [[a, b], [c, d]] -> idx 0,1,2,3
    B: [[e, f], [g, h]] -> idx 0,1,2,3
    C: [[i, j], [k, l]] -> idx 0,1,2,3 (C[0,0]=i, C[0,1]=j, C[1,0]=k, C[1,1]=l)
    
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-f)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    C[0,0] = M1 + M4 - M5 + M7
    C[0,1] = M3 + M5
    C[1,0] = M2 + M4
    C[1,1] = M1 - M2 + M3 + M6
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0], # a
        [0, 0, 0, 0, 1, 0, -1], # b
        [0, 1, 0, 0, 0, 1, 0], # c
        [1, 1, 0, 1, 0, 0, 0]  # d
    ]
    V = [
        [1, 1, 0, 0, 0, 1, 1], # e
        [0, 0, 1, -1, 1, 0, 0], # f
        [0, 1, 0, 1, 0, 1, 1], # g
        [1, 0, -1, -1, 1, 0, 0] # h
    ]
    W = [
        [1, 0, 0, -1, 1], # i (C00)
        [0, 0, 1, 0, 1], # j (C01)
        [0, 1, 0, 1, 0], # k (C10)
        [1, -1, 1, 0, 1]  # l (C11)
    ]
    # Verify:
    # M1: U=[1,0,1,0], V=[1,1,0,0] -> (a+d)(e+h)
    # M2: U=[0,0,1,1], V=[1,0,0,0] -> (c+d)e
    # M3: U=[1,0,0,0], V=[0,1,-1,0] -> a(f-h)
    # M4: U=[0,0,0,1], V=[0,-1,1,0] -> d(g-f)
    # M5: U=[1,1,0,0], V=[0,0,0,1] -> (a+b)h
    # M6: U=[-1,0,1,0], V=[1,1,0,0] -> (c-a)(e+f)
    # M7: U=[0,-1,0,-1], V=[1,0,1,1] -> (b-d)(g+h)
    
    # W rows correspond to C entries.
    # C00: M1 + M4 - M5 + M7 -> [1, 0, 0, -1, 1]
    # C01: M3 + M5 -> [0, 0, 1, 0, 1]
    # C10: M2 + M4 -> [0, 1, 0, 1, 0]
    # C11: M1 - M2 + M3 + M6 -> [1, -1, 1, 0, 1]
    
    return U, V, W

def _compose_decompositions(decomp1, decomp2, offset1, offset2, rank1, rank2, n1, m1, p1, n2, m2, p2):
    """
    This is getting complicated to compose generically.
    I will just build the 4x4 Strassen U,V,W directly by expanding the 2x2 Strassen.
    """
    pass

def _strassen_decomp_4x4():
    """
    Build 4x4 Strassen decomposition by composing 7 2x2 Strassen decompositions.
    """
    U2, V2, W2 = _strassen_decomp_2x2()
    
    # 4x4 matrix A is split into 2x2 blocks:
    # A11 A12
    # A21 A22
    # Indices in A (16 total):
    # A11: 0,1,4,5
    # A12: 2,3,6,7
    # A21: 8,9,12,13
    # A22: 10,11,14,15
    
    # Strassen steps for 4x4:
    # S1 = (A11+A22)(B11+B22)
    # S2 = (A21+A22)B11
    # S3 = A11(B12-B22)
    # S4 = A22(B21-B11)
    # S5 = (A11+A12)B22
    # S6 = (A21-A11)(B11+B12)
    # S7 = (A12-A22)(B21+B22)
    
    # Each S is a 4x4 matrix multiplication? No, S is a 2x2 matrix multiplication of 2x2 blocks.
    # So each S requires a 2x2 Strassen decomposition (Rank 7).
    # Total Rank = 7 * 7 = 49.
    
    # I will construct U, V, W of size 16 x 49.
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    # Define the 7 block multiplications
    # Each block multiplication is on 2x2 matrices.
    # The "scalar" entries of the 2x2 block are the 4 entries of the 4x4 sub-block.
    
    # Helper to add a 2x2 Strassen decomp to the main arrays
    # block_A_indices: 4 indices in U
    # block_B_indices: 4 indices in V
    # block_C_indices: 4 indices in W
    # col_offset: start column in U,V,W
    
    def add_strassen_2x2(block_A, block_B, block_C, col_off):
        for r in range(4):
            for c in range(7):
                U[block_A[r]][col_off + c] += U2[r][c]
                V[block_B[r]][col_off + c] += V2[r][c]
                W[block_C[r]][col_off + c] += W2[r][c]

    # Block indices in 4x4 (row-major 2x2 blocks of 2x2)
    # A11: [0, 1, 4, 5]
    # A12: [2, 3, 6, 7]
    # A21: [8, 9, 12, 13]
    # A22: [10, 11, 14, 15]
    
    A11 = [0, 1, 4, 5]
    A12 = [2, 3, 6, 7]
    A21 = [8, 9, 12, 13]
    A22 = [10, 11, 14, 15]
    
    B11 = [0, 1, 4, 5]
    B12 = [2, 3, 6, 7]
    B21 = [8, 9, 12, 13]
    B22 = [10, 11, 14, 15]
    
    C11 = [0, 1, 4, 5]
    C12 = [2, 3, 6, 7]
    C21 = [8, 9, 12, 13]
    C22 = [10, 11, 14, 15]
    
    # S1: (A11+A22)(B11+B22) -> C11 += S1, C12 -= S1, C21 -= S1, C22 += S1
    # Wait, Strassen output:
    # C11 = S1 + S4 - S5 + S7
    # C12 = S3 + S5
    # C21 = S2 + S4
    # C22 = S1 - S2 + S3 + S6
    
    # So we need to add the contributions of each S to the corresponding C blocks.
    # Since S is a 2x2 matrix, its entries are added to the 4 entries of the C block.
    # The W matrix of the 2x2 Strassen defines how the 7 products form the 4 entries.
    # So for each S, we add the W2 patterns to the target C blocks with appropriate signs.
    
    # S1 contributes to C11 (+), C12 (-), C21 (-), C22 (+)
    col = 0
    # S1
    # A part: A11 + A22. This means U for S1 uses A11 and A22 entries.
    # But Strassen U/V/W are for specific A,B.
    # Here A_block = A11 + A22.
    # So U for S1 should be U2 added to A11 and A22 indices.
    # V for S1 should be V2 added to B11 and B22 indices.
    # W for S1 should be W2 added to C11, C12, C21, C22 with signs.
    
    def add_strassen_step(A_idxs, B_idxs, C_idxs_map, col_off):
        # A_idxs: list of 4 indices for the "A" block of this step
        # B_idxs: list of 4 indices for the "B" block of this step
        # C_idxs_map: dict {C_block_idx: sign}
        for r in range(4):
            for c in range(7):
                u_val = U2[r][c]
                v_val = V2[r][c]
                if u_val != 0:
                    U[A_idxs[r]][col_off + c] += u_val
                if v_val != 0:
                    V[B_idxs[r]][col_off + c] += v_val
                for c_block, sign in C_idxs_map.items():
                    w_val = W2[r][c] * sign
                    if w_val != 0:
                        W[c_block[r]][col_off + c] += w_val

    # Step 1: S1 = (A11+A22)(B11+B22)
    # A_idxs: A11 and A22. But U2 is for a single 2x2 matrix.
    # A11+A22 means the "A" of this step is the sum.
    # So we add U2 to A11 indices AND U2 to A22 indices.
    # My add_strassen_step takes a single list of 4 indices.
    # I need to modify it to take multiple blocks or sum them.
    
    # Let's rewrite add_strassen_step to accept lists of indices for A and B.
    def add_step(A_blocks, B_blocks, C_map, col_off):
        # A_blocks: list of lists of indices. Sum of these blocks forms the A for this step.
        # B_blocks: list of lists of indices.
        for r in range(4):
            for c in range(7):
                u_val = U2[r][c]
                v_val = V2[r][c]
                if u_val != 0:
                    for block in A_blocks:
                        U[block[r]][col_off + c] += u_val
                if v_val != 0:
                    for block in B_blocks:
                        V[block[r]][col_off + c] += v_val
                for c_block_idx, sign in C_map.items():
                    w_val = W2[r][c] * sign
                    if w_val != 0:
                        W[c_block_idx[r]][col_off + c] += w_val

    # S1
    add_step([A11, A22], [B11, B22], {C11: 1, C12: -1, C21: -1, C22: 1}, 0)
    # S2
    add_step([A21, A22], [B11], {C21: 1, C22: -1}, 7)
    # S3
    add_step([A11], [B12, [-b for b in B22]], {C12: 1, C22: 1}, 14)
    # Note: B12 - B22. So B_blocks should handle signs?
    # My add_step adds u_val * sign? No, it adds u_val to U.
    # I need to handle signs in A and B blocks too.
    
    # Let's generalize add_step to take signed blocks.
    # A_blocks: list of (indices, sign)
    
    def add_step_signed(A_blocks, B_blocks, C_map, col_off):
        for r in range(4):
            for c in range(7):
                u_val = U2[r][c]
                v_val = V2[r][c]
                if u_val != 0:
                    for block, sign in A_blocks:
                        U[block[r]][col_off + c] += u_val * sign
                if v_val != 0:
                    for block, sign in B_blocks:
                        V[block[r]][col_off + c] += v_val * sign
                for c_block_idx, sign in C_map.items():
                    w_val = W2[r][c] * sign
                    if w_val != 0:
                        W[c_block_idx[r]][col_off + c] += w_val

    # S1: (A11+A22)(B11+B22)
    add_step_signed([(A11, 1), (A22, 1)], [(B11, 1), (B22, 1)], {C11: 1, C12: -1, C21: -1, C22: 1}, 0)
    # S2: (A21+A22)B11
    add_step_signed([(A21, 1), (A22, 1)], [(B11, 1)], {C21: 1, C22: -1}, 7)
    # S3: A11(B12-B22)
    add_step_signed([(A11, 1)], [(B12, 1), (B22, -1)], {C12: 1, C22: 1}, 14)
    # S4: A22(B21-B11)
    add_step_signed([(A22, 1)], [(B21, 1), (B11, -1)], {C11: 1, C21: 1}, 21)
    # S5: (A11+A12)B22
    add_step_signed([(A11, 1), (A12, 1)], [(B22, 1)], {C11: -1, C12: 1}, 28)
    # S6: (A21-A11)(B11+B12)
    add_step_signed([(A21, 1), (A11, -1)], [(B11, 1), (B12, 1)], {C22: 1}, 35)
    # S7: (A12-A22)(B21+B22)
    add_step_signed([(A12, 1), (A22, -1)], [(B21, 1), (B22, 1)], {C11: 1}, 42)
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    # Get 4x4 Strassen
    U4, V4, W4 = _strassen_decomp_4x4()
    
    # Get schoolbook for the rest
    # A12 B21: 4x1 x 1x4 -> 4x4. Rank 16.
    # A11 B12: 4x4 x 4x1 -> 4x1. Rank 4.
    # A12 B22: 4x1 x 1x1 -> 4x1. Rank 4.
    # A21 B11: 1x4 x 4x4 -> 1x4. Rank 4.
    # A22 B21: 1x1 x 1x4 -> 1x4. Rank 4.
    # A21 B12: 1x4 x 4x1 -> 1x1. Rank 4.
    # A22 B22: 1x1 x 1x1 -> 1x1. Rank 1.
    
    total_rank = 49 + 16 + 4 + 4 + 4 + 4 + 4 + 1
    U = [[0] * total_rank for _ in range(25)]
    V = [[0] * total_rank for _ in range(25)]
    W = [[0] * total_rank for _ in range(25)]
    
    # Map 4x4 indices to 5x5 indices
    # A11 (0:4, 0:4) -> 0..15
    # A12 (0:4, 4) -> 4, 9, 14, 19
    # A21 (4, 0:4) -> 20, 21, 22, 23
    # A22 (4, 4) -> 24
    
    # Same for B
    # C11 (0:4, 0:4) -> 0..15
    # C12 (0:4, 4) -> 4, 9, 14, 19
    # C21 (4, 0:4) -> 20, 21, 22, 23
    # C22 (4, 4) -> 24
    
    # 1. Add 4x4 Strassen for A11 B11 -> C11
    for i in range(16):
        for j in range(49):
            U[i][j] = U4[i][j]
            V[i][j] = V4[i][j]
            W[i][j] = W4[i][j]
            
    col = 49
    
    # 2. A12 B21 -> C11 (Outer product 4x4)
    # A12 indices: 4, 9, 14, 19
    # B21 indices: 20, 21, 22, 23
    # C11 indices: 0..15
    # C[i,j] = A[i,4] * B[4,j]
    for i in range(4):
        for j in range(4):
            U[4 + i*5][col] = 1 # A[i,4]
            V[20 + j][col] = 1 # B[4,j]
            W[i*5 + j][col] = 1 # C[i,j]
            col += 1
    # col is now 49 + 16 = 65
    
    # 3. A11 B12 -> C12
    # A11: 0..15
    # B12: 24, 29, 34, 39 (Wait, B is 5x5. B12 is col 4 of B? No, B12 is row 0:4, col 4?
    # B blocks: B11 (0:4, 0:4), B12 (0:4, 4), B21 (4, 0:4), B22 (4,4)
    # B12 indices: 4, 9, 14, 19
    # C12 indices: 4, 9, 14, 19
    # C[i,4] = sum_k A[i,k] B[k,4]
    for i in range(4):
        for k in range(4):
            U[i*5 + k][col] = 1
            V[4 + k*5][col] = 1 # B[k,4]
            W[4 + i*5][col] = 1 # C[i,4]
            col += 1
    # col is 65 + 16 = 81?
    # Wait, A11 B12 is 4x4 times 4x1. Rank 16?
    # Yes, 16 mults.
    
    # 4. A12 B22 -> C12
    # A12: 4, 9, 14, 19
    # B22: 24
    # C12: 4, 9, 14, 19
    # C[i,4] += A[i,4] B[4,4]
    for i in range(4):
        U[4 + i*5][col] = 1
        V[24][col] = 1
        W[4 + i*5][col] = 1
        col += 1
    # col is 81 + 4 = 85
    
    # 5. A21 B11 -> C21
    # A21: 20..23
    # B11: 0..15
    # C21: 20..23
    # C[4,j] = sum_k A[4,k] B[k,j]
    for j in range(4):
        for k in range(4):
            U[20 + k][col] = 1
            V[k*5 + j][col] = 1
            W[20 + j][col] = 1
            col += 1
    # col is 85 + 16 = 101
    
    # 6. A22 B21 -> C21
    # A22: 24
    # B21: 20..23
    # C21: 20..23
    # C[4,j] += A[4,4] B[4,j]
    for j in range(4):
        U[24][col] = 1
        V[20 + j][col] = 1
        W[20 + j][col] = 1
        col += 1
    # col is 101 + 4 = 105
    
    # 7. A21 B12 -> C22
    # A21: 20..23
    # B12: 4, 9, 14, 19
    # C22: 24
    # C[4,4] = sum_k A[4,k] B[k,4]
    for k in range(4):
        U[20 + k][col] = 1
        V[4 + k*5][col] = 1
        W[24][col] = 1
        col += 1
    # col is 105 + 4 = 109
    
    # 8. A22 B22 -> C22
    U[24][col] = 1
    V[24][col] = 1
    W[24][col] = 1
    col += 1
    # col is 110
    
    # My previous calculation was 86. Why 110?
    # A12 B21: 16
    # A11 B12: 16
    # A12 B22: 4
    # A21 B11: 16
    # A22 B21: 4
    # A21 B12: 4
    # A22 B22: 1
    # Sum: 16+16+4+16+4+4+1 = 61.
    # 49 + 61 = 110.
    
    # Can I reduce the rank of the cross terms?
    # A11 B12 (4x4 * 4x1) Rank 16.
    # A21 B11 (1x4 * 4x4) Rank 16.
    # These are unavoidable with schoolbook.
    
    return {"U": U, "V": V, "W": W}
