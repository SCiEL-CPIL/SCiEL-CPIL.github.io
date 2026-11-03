"""Tensor decomposition for 4x4 x 4x6 matrix multiplication.

Uses Kronecker product of Strassen's 2x2x2 algorithm:
- For the 4x4 x 4x4 part (first 4 columns of B): rank 49 via (2x2x2 Strassen)^2
- For the 4x4 x 4x2 part (last 2 columns of B): rank 49 via (2x2x2 Strassen) with 2x2 B-blocks
  Wait, 4x4 x 4x2: A is 4x4, B is 4x2. We can write B as two 4x1 vectors.
  4x4 x 4x1: A (4x4) * v (4x1) -> w (4x1).
  Using Strassen for 2x2 blocks: A has 4 blocks of 2x2, v has 2 blocks of 2x1.
  w_block_j = sum_i A_{ij} v_i where A_{ij} is 2x2, v_i is 2x1.
  Each 2x2 * 2x1 -> 2x1 product. 
  2x2 * 2x1: Let A = [[a,b],[c,d]], v = [x,y]. 
  Result = [ax+by, cx+dy]. 
  This is just 4 multiplications, rank 4.
  So 4x4 * 4x1 = 4 blocks in A row, 2 blocks in v column.
  w_0 = A_00*v_0 + A_01*v_1
  w_1 = A_10*v_0 + A_11*v_1
  Each term is 2x2 * 2x1 = rank 4.
  Total 4 terms * rank 4 = rank 16 for 4x4 * 4x1.
  So 4x4 * 4x2 = two such products = rank 32.

Total rank = 49 + 32 = 81.
"""

DIMENSIONS = [4, 4, 6]


def construct():
    """
    Returns a dictionary with 'U', 'V', 'W' factor matrices.
    
    U: shape (16, rank) - corresponds to A entries (4x4 = 16)
    V: shape (24, rank) - corresponds to B entries (4x6 = 24)  
    W: shape (16, rank) - corresponds to output entries (4x4 = 16)
    
    Note: The task says W[k*n+i, r] for output factor with shapes (p*n, rank)
    p*n = 6*4 = 24? Wait, let me re-read.
    
    "output factor follows the official trace convention W[k*n+i,r]"
    "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    
    n=4, m=4, p=6.
    U: (n*m, rank) = (16, rank) - A is 4x4
    V: (m*p, rank) = (24, rank) - B is 4x6
    W: (p*n, rank) = (24, rank) - Output is 4x6? 
    
    Wait, matrix multiplication: (4x4) * (4x6) = (4x6).
    Output is 4 rows, 6 columns. p*n = 6*4 = 24. Yes.
    
    The output C[i,k] = sum_j A[i,j] B[j,k].
    W convention: W[k*n + i, r] so W index is k*4 + i where k is column (0..5), i is row (0..3).
    So W has 24 rows.
    """
    n, m, p = 4, 4, 6
    # n=4 rows of A, m=4 cols of A / rows of B, p=6 cols of B
    # A: n x m = 4x4
    # B: m x p = 4x6
    # C: n x p = 4x6
    # U: n*m = 16 rows (A indices: i*m+j = i*4+j)
    # V: m*p = 24 rows (B indices: j*p+k = j*6+k)
    # W: p*n = 24 rows (C indices: k*n+i = k*4+i)
    
    # Strassen 2x2x2 vectors
    # For 2x2 * 2x2 -> 2x2:
    # C_00 = M0 - M1 + M2 + M3
    # C_01 = M1 + M4
    # C_10 = M2 + M5
    # C_11 = M0 - M3 + M4 + M5
    # where:
    # M0 = (a00+a11)(b00+b11)
    # M1 = (a10+a11)*b00
    # M2 = a00*(b10-b11)
    # M3 = a11*(b01-b00)
    # M4 = (a00+a01)*b11
    # M5 = (a10-a00)*(b00+b01)
    # M6 = (a01-a11)*(b10+b11)
    
    # For each M_r, we have u (for A), v (for B), w (for C) vectors of length 4.
    # A indices: 00,01,10,11 -> 0,1,2,3
    # B indices: 00,01,10,11 -> 0,1,2,3
    # C indices: 00,01,10,11 -> 0,1,2,3
    
    strassen = [
        # M0
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, -1]),  # C00, C11
        # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, -1, 0]),   # C01, C10
        # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 1, 0]),   # C10
        # M3
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 0, -1]),  # C00, C11
        # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]),    # C01, C11
        # M5
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 1, 0]),   # C10
        # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, -1, 0, 0]),  # C01 neg, C10 neg
    ]
    
    # Now build 4x4 x 4x4 decomposition via Kronecker product.
    # 4x4 A = 4 blocks of 2x2: A_00, A_01, A_10, A_11
    # 4x4 B = 4 blocks of 2x2: B_00, B_01, B_10, B_11
    # 4x4 C = 4 blocks of 2x2: C_00, C_01, C_10, C_11
    
    # For the outer 2x2x2 Strassen on the block level:
    # C_00_block = M0_block - M1_block + M2_block + M3_block
    # C_01_block = M1_block + M4_block
    # C_10_block = M2_block + M5_block
    # C_11_block = M0_block - M3_block + M4_block + M5_block
    
    # Each M_r_block is a 2x2x2 Strassen product of the inner blocks.
    
    # So total rank = 7 (outer) * 7 (inner) = 49 for 4x4x4.
    
    # Let's construct the 4x4x4 part.
    # Outer Strassen vectors (same as above but for block indices):
    # Outer A block indices: 00,01,10,11 -> corresponds to A blocks
    # Outer B block indices: 00,01,10,11
    # Outer C block indices: 00,01,10,11
    
    # For each outer M_r, we combine with each inner M_s.
    # Outer M_r gives weights for which A-blocks and B-blocks to multiply.
    # Inner M_s gives the actual 2x2x2 Strassen within those blocks.
    
    # A full 4x4 index i*4+j where i,j in 0..3.
    # Block row = i//2, block col = j//2.
    # Inner row = i%2, inner col = j%2.
    
    # For a given outer (r) and inner (s):
    # U entry at A[i,j] = outer_u[r][block_i][block_j] * inner_u[s][inner_i][inner_j]
    # Wait, no. The outer product selects which A-blocks and B-blocks participate.
    # Actually, for outer M_r, the A-block combination is outer_u[r] applied to A-blocks.
    # For inner M_s, the actual values are inner_u[s] applied within each selected block.
    
    # Let me think differently. 
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Split into blocks: i = 2*I + i2, j = 2*J + j2, k = 2*K + k2
    # C[2I+i2, 2K+k2] = sum_{J,j2} A[2I+i2, 2J+j2] B[2J+j2, 2K+k2]
    #                   = sum_J (A_block[I,J] * B_block[J,K])_{i2,k2}
    
    # For fixed I,J,K, (A_block[I,J] * B_block[J,K]) is a 2x2x2 tensor.
    # Sum over J gives the 4x4x4 tensor as sum of 4 such 2x2x2 tensors (one per J).
    # But Strassen reduces the outer sum over J (which is a 2x2x2 "matrix multiplication" of blocks).
    
    # The outer 2x2x2 has:
    # A-blocks: A_00, A_01, A_10, A_11 (I,J in 0,1)
    # B-blocks: B_00, B_01, B_10, B_11 (J,K in 0,1)
    # C-blocks: C_00, C_01, C_10, C_11 (I,K in 0,1)
    
    # Outer Strassen computes C_blocks = f(A_blocks, B_blocks) with 7 multiplications.
    # Each multiplication is of the form (linear combo of A-blocks) * (linear combo of B-blocks)
    # But the "multiplication" of two 2x2 matrices is itself a 2x2x2 tensor, which we
    # decompose with inner Strassen (7 more multiplications).
    
    # So for outer M_r and inner M_s:
    # U[A_index] = outer_u[r][A_block] * inner_u[s][A_inner]
    # where A_index = (2*I+i2)*4 + (2*J+j2) = (2*I+i2)*4 + 2*J + j2
    # A_block index for outer: I*2+J (0,1,2,3 for 00,01,10,11)
    # A_inner index for inner: i2*2+j2 (0,1,2,3)
    
    # Similarly for V and W.
    
    # Let's build the 49 rank terms for 4x4x4.
    terms_444 = []
    for r in range(7):
        outer_u = strassen[r][0]  # [u00, u01, u10, u11] for A blocks
        outer_v = strassen[r][1]  # [v00, v01, v10, v11] for B blocks
        outer_w = strassen[r][2]  # [w00, w01, w10, w11] for C blocks
        for s in range(7):
            inner_u = strassen[s][0]  # [u00, u01, u10, u11] for inner A
            inner_v = strassen[s][1]  # [v00, v01, v10, v11] for inner B
            inner_w = strassen[s][2]  # [w00, w01, w10, w11] for inner C
            
            U_vec = [0.0] * 16
            V_vec = [0.0] * 24
            W_vec = [0.0] * 24  # Only first 16 used for 4x4x4 part
            
            for I in range(2):
                for J in range(2):
                    block_idx = I * 2 + J
                    for i2 in range(2):
                        for j2 in range(2):
                            inner_idx = i2 * 2 + j2
                            A_idx = (2 * I + i2) * 4 + (2 * J + j2)
                            U_vec[A_idx] += outer_u[block_idx] * inner_u[inner_idx]
            
            for J in range(2):
                for K in range(2):
                    block_idx = J * 2 + K
                    for j2 in range(2):
                        for k2 in range(2):
                            inner_idx = j2 * 2 + k2
                            B_idx = (2 * J + j2) * 6 + (2 * K + k2)
                            V_vec[B_idx] += outer_v[block_idx] * inner_v[inner_idx]
            
            for I in range(2):
                for K in range(2):
                    block_idx = I * 2 + K
                    for i2 in range(2):
                        for k2 in range(2):
                            inner_idx = i2 * 2 + k2
                            C_idx = (2 * K + k2) * 4 + (2 * I + i2)
                            W_vec[C_idx] += outer_w[block_idx] * inner_w[inner_idx]
            
            terms_444.append((U_vec, V_vec, W_vec))
    
    # Now for 4x4 x 4x2 part (last 2 columns of B, indices k=4,5).
    # C[i,k] for k=4,5.
    # B is 4x2 for this part. B[j,4] and B[j,5].
    # C[i,4] = sum_j A[i,j] B[j,4]
    # C[i,5] = sum_j A[i,j] B[j,5]
    
    # Split A into 2x2 blocks: A[2I+i2, 2J+j2]
    # B vector: B[2J+j2, 4] and B[2J+j2, 5]
    # C vector: C[2I+i2, 4] and C[2I+i2, 5]
    
    # For each pair of blocks (I,J):
    # C_block_col[2I+i2, 4] = sum_{J,j2} A[2I+i2,2J+j2] B[2J+j2,4]
    # C_block_col[2I+i2, 5] = sum_{J,j2} A[2I+i2,2J+j2] B[2J+j2,5]
    
    # This is: for each I, C_col_I = sum_J (A_block[I,J] * v_block[J])
    # where v_block[J] = [B[2J,4], B[2J+1,4], B[2J,5], B[2J+1,5]]
    # and result is [C[2I,4], C[2I+1,4], C[2I,5], C[2I+1,5]]
    
    # 2x2 matrix * 2x2 matrix (as two column vectors) -> 2x2 matrix
    # This is just 2x2 * 2x2 tensor, rank 7 via Strassen!
    # But wait, the "B" here is really a 2x2 matrix formed from the two columns.
    
    # So for 4x4 * 4x2:
    # A has 4 blocks (2x2 grid): A_00, A_01, A_10, A_11
    # B has 2 blocks vertically: B_top, B_bottom, each 2x2 (2 rows x 2 cols)
    # C has 2 blocks vertically: C_top, C_bottom, each 2x2
    
    # C_top = A_00 * B_top + A_01 * B_bottom
    # C_bottom = A_10 * B_top + A_11 * B_bottom
    
    # This is a 2x2x2 "multiplication" where:
    # "A" = [A_00, A_01, A_10, A_11] (4 blocks)
    # "B" = [B_top, B_bottom] but each "B" element is 2x2, so we need to handle this.
    
    # Actually, let's think of it as: for each J (0,1), we have a 2x2 * 2x2 product.
    # C_block[I] = sum_J A_block[I,J] * B_block[J]
    
    # This is a 2x2 matrix (in I) multiplied by a 2x2 matrix (in J, with 2x2 entries).
    # The "multiplication" of two 2x2 matrices with 2x2 entries is equivalent to
    # a 4x4x4 tensor if we flatten, but here B and C have different structure.
    
    # Let me just use schoolbook for this part to be safe: rank 32.
    # Or better: for each J, the 2x2 * 2x2 -> 2x2 has rank 7.
    # C_block[I] = sum_J A_block[I,J] * B_block[J]
    # For fixed I, this is sum over J=0,1 of two 2x2*2x2 products.
    # Using Strassen for each: rank 7 per (I,J) pair.
    # Total: 2 (I values) * 2 (J values) * 7 = 28? No, that's not right either.
    
    # Actually, for 4x4 * 4x2:
    # We can view it as 4 independent 4x1 * 4x1 -> 4x1 products? No.
    # Let's just do: for each k in {4,5}, compute C[:,k] = A * B[:,k].
    # Each is 4x4 * 4x1 -> 4x1.
    # 4x4 * 4x1: split into 2x2 blocks.
    # C_block[I] = sum_J A_block[I,J] * v_block[J]
    # 2x2 * 2x1 -> 2x1: rank 4 (schoolbook) or can we do better?
    # 2x2 * 2x1: [[a,b],[c,d]] * [x,y] = [ax+by, cx+dy]. Rank 4.
    # So 4x4 * 4x1: 2*2*4 = 16 rank terms.
    # 4x4 * 4x2: 2 * 16 = 32 rank terms.
    
    # Total rank: 49 + 32 = 81.
    
    # Build 32 terms for 4x4 x 4x2.
    terms_442 = []
    for k_offset in [4, 5]:
        for I in range(2):
            for J in range(2):
                for a2 in range(2):
                    for b2 in range(2):
                        # A[2I+a2, 2J+b2] * B[2J+b2, k_offset]
                        A_idx = (2 * I + a2) * 4 + (2 * J + b2)
                        B_idx = (2 * J + b2) * 6 + k_offset
                        C_idx = k_offset * 4 + (2 * I + a2)
                        
                        U_vec = [0.0] * 16
                        V_vec = [0.0] * 24
                        W_vec = [0.0] * 24
                        
                        U_vec[A_idx] = 1.0
                        V_vec[B_idx] = 1.0
                        W_vec[C_idx] = 1.0
                        
                        terms_442.append((U_vec, V_vec, W_vec))
    
    # Combine all terms
    all_terms = terms_444 + terms_442
    rank = len(all_terms)
    
    U = [[0.0] * rank for _ in range(16)]
    V = [[0.0] * rank for _ in range(24)]
    W = [[0.0] * rank for _ in range(24)]
    
    for r_idx, (u, v, w) in enumerate(all_terms):
        for i in range(16):
            if u[i] != 0:
                U[i][r_idx] = u[i]
        for i in range(24):
            if v[i] != 0:
                V[i][r_idx] = v[i]
        for i in range(24):
            if w[i] != 0:
                W[i][r_idx] = w[i]
    
    # Convert to integer where possible
    def to_int_if_close(val):
        if abs(val - round(val)) < 1e-9:
            return int(round(val))
        return val
    
    for i in range(16):
        for r_idx in range(rank):
            U[i][r_idx] = to_int_if_close(U[i][r_idx])
    for i in range(24):
        for r_idx in range(rank):
            V[i][r_idx] = to_int_if_close(V[i][r_idx])
    for i in range(24):
        for r_idx in range(rank):
            W[i][r_idx] = to_int_if_close(W[i][r_idx])
    
    return {"U": U, "V": V, "W": W}
