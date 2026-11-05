import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Partition indices into S1={0,1,2,3} and S2={4}
    # S1 size 4, S2 size 1
    
    # We will build U, V, W with total rank R
    # We need to calculate the total rank first to allocate memory
    
    # Block ranks:
    # 1. 4x4x4: 49
    # 2. 4x4x1: 16
    # 3. 4x1x4: 1
    # 4. 4x1x1: 1
    # 5. 1x4x4: 4
    # 6. 1x4x1: 4
    # 7. 1x1x4: 4
    # 8. 1x1x1: 1
    
    # Total Rank = 49 + 16 + 1 + 1 + 4 + 4 + 4 + 1 = 80
    
    R = 80
    U = [[0] * R for _ in range(n * m)] # 25 x 80
    V = [[0] * R for _ in range(m * p)] # 25 x 80
    W = [[0] * R for _ in range(p * n)] # 25 x 80
    
    col_offset = 0
    
    # Helper to add a block decomposition
    # block_n, block_m, block_p: sizes
    # start_i, start_j, start_k: starting indices in global 0..4
    # rank: rank of this block
    # U_block, V_block, W_block: matrices of shapes (bn*bm, rank), (bm*bp, rank), (bp*bn, rank)
    
    def add_block(bn, bm, bp, si, sj, sk, U_b, V_b, W_b):
        nonlocal col_offset
        rank_b = len(U_b[0])
        for r in range(rank_b):
            # U
            for i in range(bn):
                for j in range(bm):
                    val = U_b[i * bm + j][r]
                    if val != 0:
                        U[(si + i) * m + (sj + j)][col_offset + r] = val
            # V
            for j in range(bm):
                for k in range(bp):
                    val = V_b[j * bp + k][r]
                    if val != 0:
                        V[(sj + j) * p + (sk + k)][col_offset + r] = val
            # W
            for k in range(bp):
                for i in range(bn):
                    val = W_b[k * bn + i][r]
                    if val != 0:
                        W[(sk + k) * n + (si + i)][col_offset + r] = val
        col_offset += rank_b

    # 1. 4x4x4 block (Rank 49)
    # Use Strassen for 4x4
    # Strassen for 2x2 has rank 7.
    # 4x4 is 2x2 blocks of 2x2.
    # We need to generate the rank 49 decomposition.
    
    # Let's generate Strassen 2x2 decomposition first
    # U2, V2, W2 for 2x2x2
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Strassen products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Indices for 2x2:
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11
    
    # We define the 7 rank-1 tensors for 2x2
    # Each tensor is defined by U, V, W vectors of length 4
    
    # M1: (A00+A11)(B00+B11)
    # C00 += M1, C01 += M1, C10 += M1, C11 += M1
    # U: A00=1, A11=1
    # V: B00=1, B11=1
    # W: C00=1, C01=1, C10=1, C11=1
    
    def set_rank1(idx, u_vec, v_vec, w_vec):
        for i in range(4):
            U2[i][idx] = u_vec[i]
            V2[i][idx] = v_vec[i]
            W2[i][idx] = w_vec[i]
            
    # M1
    set_rank1(0, [1,0,0,1], [1,0,0,1], [1,1,1,1])
    # M2
    set_rank1(1, [0,0,1,1], [1,0,0,0], [0,0,1,1]) # (A10+A11)B00 -> C10+=M2, C11+=M2
    # M3
    set_rank1(2, [1,0,0,0], [0,0,1,-1], [0,1,0,0]) # A00(B10-B11) -> C01+=M3
    # M4
    set_rank1(3, [0,0,0,1], [0,1,-1,0], [0,0,0,1]) # A11(B01-B00) -> C11+=M4
    # M5
    set_rank1(4, [1,1,0,0], [0,0,0,1], [1,1,0,0]) # (A00+A01)B11 -> C00+=M5, C01+=M5
    # M6
    set_rank1(5, [-1,0,1,0], [1,1,0,0], [0,1,0,0]) # (A10-A00)(B00+B01) -> C01+=M6
    # M7
    set_rank1(6, [0,-1,0,1], [0,0,1,1], [0,0,1,1]) # (A01-A11)(B10+B11) -> C10+=M7, C11+=M7
    
    # Now build 4x4 from 2x2 Strassen
    # 4x4 indices: 0..3
    # Blocks:
    # A11 (0,1 x 0,1), A12 (0,1 x 2,3)
    # A21 (2,3 x 0,1), A22 (2,3 x 2,3)
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # For each of the 7 Strassen steps, we have a 2x2 multiplication.
    # Each 2x2 multiplication uses 7 rank-1 tensors.
    # Total 49.
    
    # Map global 4x4 indices to local 2x2 indices for each block
    # Block (bi, bj) where bi,bj in {0,1}
    # Global i = 2*bi + li, j = 2*bj + lj
    
    for s in range(7):
        # Determine which A and B blocks are involved in Strassen step s
        # We need the linear combinations of A and B blocks for step s
        # This is complex to generate dynamically. 
        # Instead, we can just hardcode the 49 rank-1 tensors for 4x4?
        # Or use a recursive function.
        
        # Let's use a recursive approach.
        # We already have U2, V2, W2 for 2x2.
        # For 4x4, we apply Strassen logic to the 2x2 blocks.
        
        # Strassen step s involves specific linear combinations of the 4 A-blocks and 4 B-blocks.
        # Let A_blocks = [A00, A01, A10, A11] (each 2x2)
        # Let B_blocks = [B00, B01, B10, B11]
        
        # We need to define the coefficients for A_blocks and B_blocks for each step s.
        # Then for each rank-1 component of the 2x2 multiplication (from U2, V2, W2),
        # we expand it into 4x4.
        
        # Coefficients for Strassen steps (s=0..6)
        # A coeffs (for A00, A01, A10, A11)
        # B coeffs (for B00, B01, B10, B11)
        
        a_coeffs = [
            [1, 0, 0, 1], # M1: A00+A11
            [0, 0, 1, 1], # M2: A10+A11
            [1, 0, 0, 0], # M3: A00
            [0, 0, 0, 1], # M4: A11
            [1, 1, 0, 0], # M5: A00+A01
            [-1, 0, 1, 0], # M6: A10-A00
            [0, -1, 0, 1]  # M7: A01-A11
        ]
        b_coeffs = [
            [1, 0, 0, 1], # M1: B00+B11
            [1, 0, 0, 0], # M2: B00
            [0, 0, 1, -1], # M3: B10-B11
            [0, 1, -1, 0], # M4: B01-B00
            [0, 0, 0, 1], # M5: B11
            [1, 1, 0, 0], # M6: B00+B01
            [0, 0, 1, 1]  # M7: B10+B11
        ]
        
        # For each step s, we compute the product of the linear combinations.
        # The product is a 2x2 matrix multiplication of the resulting 2x2 blocks.
        # We use the rank-7 decomposition of 2x2 multiplication.
        
        for r2 in range(7):
            # U2, V2, W2 for this rank-1 component
            # U2 has values for A00, A01, A10, A11 (local 2x2)
            # V2 has values for B00, B01, B10, B11
            # W2 has values for C00, C01, C10, C11
            
            # We need to map this to 4x4.
            # The "A" in the 2x2 multiplication is the linear combination of A-blocks.
            # A_local[li, lj] = sum_{bi,bj} a_coeffs[s][bi*2+bj] * A_block[bi,bj][li, lj]
            
            # So for global index i = 2*bi + li, j = 2*bj + lj:
            # U4[i, j, r4] = U2[li*2+lj, r2] * a_coeffs[s][bi*2+bj]
            
            # Similarly for V and W.
            
            r4 = s * 7 + r2
            
            for bi in range(2):
                for bj in range(2):
                    coeff_a = a_coeffs[s][bi * 2 + bj]
                    coeff_b = b_coeffs[s][bi * 2 + bj]
                    
                    if coeff_a == 0 and coeff_b == 0:
                        continue
                        
                    for li in range(2):
                        for lj in range(2):
                            gi = 2 * bi + li
                            gj = 2 * bj + lj
                            
                            # U4
                            val_u = U2[li * 2 + lj][r2] * coeff_a
                            if val_u != 0:
                                U4[gi * 4 + gj][r4] = val_u
                            
                            # V4
                            val_v = V2[li * 2 + lj][r2] * coeff_b
                            if val_v != 0:
                                V4[gi * 4 + gj][r4] = val_v
                            
                            # W4
                            # W2 is for C blocks.
                            # C_local[li, lj] corresponds to C_block[bi, bj]
                            val_w = W2[li * 2 + lj][r2]
                            if val_w != 0:
                                W4[gi * 4 + gj][r4] = val_w
                                
    add_block(4, 4, 4, 0, 0, 0, U4, V4, W4)

    # 2. 4x4x1 block (Rank 16)
    # i in 0..3, j in 0..3, k=4
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # Rank 16: one for each (i, j)
    U_b = [[0]*16 for _ in range(16)]
    V_b = [[0]*16 for _ in range(4)]
    W_b = [[0]*16 for _ in range(4)]
    
    for i in range(4):
        for j in range(4):
            r = i * 4 + j
            U_b[i * 4 + j][r] = 1
            V_b[j * 1 + 0][r] = 1 # j in 0..3, k=0 (local)
            W_b[0 * 4 + i][r] = 1 # k=0 (local), i in 0..3
            
    add_block(4, 4, 1, 0, 0, 4, U_b, V_b, W_b)

    # 3. 4x1x4 block (Rank 1)
    # i in 0..3, j=4, k in 0..3
    # C[i, k] = A[i, 4] B[4, k]
    # Rank 1
    U_b = [[0]*1 for _ in range(4)]
    V_b = [[0]*1 for _ in range(4)]
    W_b = [[0]*1 for _ in range(4)]
    
    for i in range(4):
        U_b[i * 1 + 0][0] = 1
        V_b[0 * 4 + 0][0] = 1
        W_b[0 * 4 + i][0] = 1 # Wait, W is p x n. p=4, n=4.
        # W index: k * n + i. k in 0..3, i in 0..3.
        # We need W[k, i] = 1 for all k, i?
        # C[i, k] = A[i, 4] B[4, k].
        # U[i, 0] = A[i, 4] -> 1
        # V[4, 0] = B[4, k]? No, V is m x p. m=1, p=4.
        # V[j, k]. j=0 (local).
        # We need V[0, k] = 1 for all k.
        # W[k, i] = 1 for all k, i.
        
    U_b = [[1] for _ in range(4)]
    V_b = [[1, 1, 1, 1]]
    W_b = [[1, 1, 1, 1] for _ in range(4)]
    
    add_block(4, 1, 4, 0, 4, 0, U_b, V_b, W_b)

    # 4. 4x1x1 block (Rank 1)
    # i in 0..3, j=4, k=4
    # C[i, 4] = A[i, 4] B[4, 4]
    U_b = [[1] for _ in range(4)]
    V_b = [[1]]
    W_b = [[1] for _ in range(4)]
    
    add_block(4, 1, 1, 0, 4, 4, U_b, V_b, W_b)

    # 5. 1x4x4 block (Rank 4)
    # i=4, j in 0..3, k in 0..3
    # C[4, k] = sum_j A[4, j] B[j, k]
    # Rank 4: one for each j
    U_b = [[0]*4 for _ in range(4)]
    V_b = [[0]*4 for _ in range(16)]
    W_b = [[0]*4 for _ in range(4)]
    
    for j in range(4):
        r = j
        U_b[0 * 4 + j][r] = 1
        V_b[j * 4 + 0][r] = 1 # V is m x p. m=4, p=4.
        # V[j, k]. We need V[j, k] = 1 for all k?
        # C[k] = sum_j A[j] B[j, k].
        # Term j: A[j] * B[j, k].
        # U[0, j] = A[j] -> 1
        # V[j, k] = B[j, k] -> 1 for all k
        # W[k, 0] = 1 for all k
        
        for k in range(4):
            V_b[j * 4 + k][r] = 1
            W_b[k * 1 + 0][r] = 1
            
    add_block(1, 4, 4, 4, 0, 0, U_b, V_b, W_b)

    # 6. 1x4x1 block (Rank 4)
    # i=4, j in 0..3, k=4
    # C[4, 4] = sum_j A[4, j] B[j, 4]
    # Rank 4: one for each j
    U_b = [[0]*4 for _ in range(4)]
    V_b = [[0]*4 for _ in range(4)]
    W_b = [[0]*4 for _ in range(1)]
    
    for j in range(4):
        r = j
        U_b[0 * 4 + j][r] = 1
        V_b[j * 1 + 0][r] = 1
        W_b[0 * 1 + 0][r] = 1
        
    add_block(1, 4, 1, 4, 0, 4, U_b, V_b, W_b)

    # 7. 1x1x4 block (Rank 4)
    # i=4, j=4, k in 0..3
    # C[4, k] = A[4, 4] B[4, k]
    # Rank 4: one for each k
    U_b = [[0]*4 for _ in range(1)]
    V_b = [[0]*4 for _ in range(4)]
    W_b = [[0]*4 for _ in range(4)]
    
    for k in range(4):
        r = k
        U_b[0 * 1 + 0][r] = 1
        V_b[0 * 4 + k][r] = 1
        W_b[k * 1 + 0][r] = 1
        
    add_block(1, 1, 4, 4, 4, 0, U_b, V_b, W_b)

    # 8. 1x1x1 block (Rank 1)
    # i=4, j=4, k=4
    # C[4, 4] = A[4, 4] B[4, 4]
    U_b = [[1]]
    V_b = [[1]]
    W_b = [[1]]
    
    add_block(1, 1, 1, 4, 4, 4, U_b, V_b, W_b)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
