"""
Tensor decomposition for 4x4 * 4x6 matrix multiplication.

Strategy: decompose the output columns (the 6 columns of the 4x6 result) into
a 4-column part and a 2-column part.

For the 4x4 part, use a Strassen-style rank-49 decomposition.
For the 4x2 part, use schoolbook rank 32.

Total rank = 49 + 32 = 81.

The entries are integers (±1, ±2), satisfying the half-integer constraint.
"""

DIMENSIONS = [4, 4, 6]

def _strassen_2x2x2():
    """
    Return the 7 rank-1 terms for 2x2x2 Strassen.
    Each term is (u_vec, v_vec, w_vec) where each vec is length 4.
    u_vec[i] = U[i, term], v_vec[j] = V[j, term], w_vec[k] = W[k, term]
    for the 2x2x2 tensor with indices (i, j, k) in {0,1}^3.
    
    The 2x2x2 multiplication tensor T[i,j,k] = 1 for all (i,j,k)
    represents C[i,k] = sum_j A[i,j] B[j,k].
    
    Strassen's 7 products:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11)B00
    M3 = A00(B10 + B11)
    M4 = A11(B01 - B11)
    M5 = (A00 + A01)B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    Then:
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # For each Strassen product, we need to express it as a rank-1 tensor.
    # A product (sum_{i,j} a_ij A[i,j]) * (sum_{j,k} b_jk B[j,k])
    # contributes to C[i,k] with coefficient a_ij * b_jk for each (i,j,k).
    # So the rank-1 tensor has U[i] = a_ij, V[j] = b_jk... wait.
    # Actually for a rank-1 decomposition of the tensor:
    # T = sum_r U_r ⊗ V_r ⊗ W_r
    # where T[i,j,k] = sum_r U_r[i] * V_r[j] * W_r[k]
    # 
    # For Strassen, each M_r is a product of a linear form in A and a linear form in B.
    # M_r = (sum_{i,j} u_{r,i,j} A[i,j]) * (sum_{j,k} v_{r,j,k} B[j,k])
    # This contributes to C[i,k]: for each j, the coefficient is u_{r,i,j} * v_{r,j,k}.
    # So in the tensor decomposition: U_r[i] should capture the A part, V_r[j] the B part...
    # But it's not separable that simply. Let me think again.
    #
    # The multiplication tensor T_{ijk} for 2x2x2: T[i,j,k] = 1 if we compute C[i,k] += A[i,j]*B[j,k].
    # Actually the standard tensor for matmul: the structure tensor is T[i,k,j] = 1 (or T[i,j,k]).
    # Let's use T[i,j,k] = 1 for all i,j,k in {0,1}, representing the 8 scalar multiplications.
    #
    # Strassen decomposes this into 7 rank-1 tensors.
    # Each rank-1 term: for product M_r = (L_A^r)(A) * (L_B^r)(B),
    # the contribution to C is: C += (L_A^r)(A) ⊗ (L_B^r)(B)
    # where (L_A^r)(A) is a 2x2 matrix and (L_B^r)(B) is a 2x2 matrix? No.
    # 
    # L_A^r is a linear form on the 4 entries of A, giving a scalar.
    # L_B^r is a linear form on the 4 entries of B, giving a scalar.
    # Then M_r is a scalar, and C is reconstructed as linear combinations of M_r's.
    #
    # So the rank-1 tensor for term r has:
    # U_r[i,j] = coefficient of A[i,j] in L_A^r  (flattened: U_r[i*2+j])
    # V_r[j,k] = coefficient of B[j,k] in L_B^r  (flattened: V_r[j*2+k])
    # W_r[i,k] = coefficient of M_r in C[i,k]   (flattened: W_r[k*2+i] per task convention W[k*n+i])
    #
    # Let me define the 7 Strassen terms:
    
    terms = []
    # M1 = (A00 + A11)(B00 + B11)
    # L_A: A00 + A11 -> [1, 0, 0, 1] for indices (00, 01, 10, 11)
    # L_B: B00 + B11 -> [1, 0, 0, 1] for indices (00, 01, 10, 11) where B[j,k] -> j*2+k
    # C: C00 += M1, C11 += M1 -> W[0*2+0]=1, W[1*2+1]=1 -> W indices: k*2+i
    # W[0,0]=1 (k=0,i=0), W[1,1]=1 (k=1,i=1)
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # M2 = (A10 + A11)B00
    # L_A: A10 + A11 -> [0, 0, 1, 1]
    # L_B: B00 -> [1, 0, 0, 0]
    # C: C10 += M2, C11 -= M2 -> W[k*2+i]: C[1,0]: k=0,i=1 -> W[1]=1; C[1,1]: k=1,i=1 -> W[3]=-1
    terms.append(([0,0,1,1], [1,0,0,0], [0,1,-1,0]))
    
    # M3 = A00(B10 + B11)
    # L_A: A00 -> [1, 0, 0, 0]
    # L_B: B10 + B11 -> [0, 0, 1, 1]
    # C: C01 += M3, C11 += M3 -> W[k*2+i]: C[0,1]: k=1,i=0 -> W[2]=1; C[1,1]: k=1,i=1 -> W[3]=1
    terms.append(([1,0,0,0], [0,0,1,1], [0,0,1,1]))
    
    # M4 = A11(B01 - B11)
    # L_A: A11 -> [0, 0, 0, 1]
    # L_B: B01 - B11 -> [0, 1, 0, -1]
    # C: C00 += M4, C10 += M4 -> W[0]=1 (k=0,i=0), W[1]=1 (k=0,i=1)
    terms.append(([0,0,0,1], [0,1,0,-1], [1,1,0,0]))
    
    # M5 = (A00 + A01)B11
    # L_A: A00 + A01 -> [1, 1, 0, 0]
    # L_B: B11 -> [0, 0, 0, 1]
    # C: C00 -= M5, C01 += M5 -> W[0]=-1 (k=0,i=0), W[2]=1 (k=1,i=0)
    terms.append(([1,1,0,0], [0,0,0,1], [-1,0,1,0]))
    
    # M6 = (A10 - A00)(B00 + B01)
    # L_A: A10 - A00 -> [-1, 0, 1, 0]
    # L_B: B00 + B01 -> [1, 1, 0, 0]
    # C: C11 += M6 -> W[3]=1 (k=1,i=1)
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    
    # M7 = (A01 - A11)(B10 + B11)
    # L_A: A01 - A11 -> [0, 1, 0, -1]
    # L_B: B10 + B11 -> [0, 0, 1, 1]
    # C: C00 += M7 -> W[0]=1 (k=0,i=0)
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    
    return terms

def _build_strassen_4x4x4():
    """
    Build the rank-49 decomposition for 4x4 * 4x4 -> 4x4.
    Uses block Strassen: partition 4x4 into four 2x2 blocks.
    Apply Strassen at block level (7 block products), each requiring 7 scalar products.
    
    Actually, a simpler approach: just use 49 individual rank-1 tensors where
    we apply the 2x2x2 Strassen pattern independently to each of the 4 
    2x2x2 sub-tensors? No, that would give 4*7=28, which is not correct for 4x4x4.
    
    Let me use a different approach: the 4x4x4 multiplication tensor can be
    decomposed by noting that 4=2+2. Use the fact that:
    
    If we have a rank-R decomposition for n x m * m x p, we can use it for
    larger problems by block decomposition.
    
    For 4x4*4x4: partition into 2x2 blocks of size 2x2.
    A = [[A00, A01], [A10, A11]] where each Ajk is 2x2.
    B = [[B00, B10], [B01, B11]] where each Bjk is 2x2. (B[j,k] block)
    C = [[C00, C01], [C10, C11]] where each Cik is 2x2.
    
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    Using Strassen at block level:
    M1 = (A00+A11)*(B00+B11)  -> 2x2 * 2x2 product
    M2 = (A10+A11)*B00        -> 2x2 * 2x2
    M3 = A00*(B10+B11)        -> 2x2 * 2x2
    M4 = A11*(B01-B11)        -> 2x2 * 2x2
    M5 = (A00+A01)*B11        -> 2x2 * 2x2
    M6 = (A10-A00)*(B00+B01)  -> 2x2 * 2x2
    M7 = (A01-A11)*(B10+B11)  -> 2x2 * 2x2
    
    Each Mik is a 2x2 matrix product, which can be computed with rank 7 (Strassen).
    So total rank = 7 * 7 = 49.
    
    To construct U, V, W:
    For each block-level term r (0..6) and each intra-block term s (0..6):
    We have a combined rank-1 tensor.
    
    U indices: (i, j) for 4x4 A, flattened as i*4+j.
    V indices: (j, k) for 4x4 B, flattened as j*4+k.
    W indices: (k, i) for 4x4 C, flattened as k*4+i (per task: W[k*n+i]).
    
    Block indices: i = 2*ib + i2, j = 2*jb + j2, k = 2*kb + k2
    where ib,jb,kb in {0,1} are block indices, i2,j2,k2 in {0,1} are intra-block.
    
    For block term r and intra-block term s:
    U[i*4+j] = U_block_r[ib, jb] * U_intra_s[i2, j2]
    V[j*4+k] = V_block_r[jb, kb] * V_intra_s[j2, k2]
    W[k*4+i] = W_block_r[kb, ib] * W_intra_s[k2, i2]
    
    where U_block_r, V_block_r, W_block_r come from the 2x2x2 Strassen (with 2x2 "matrices" being scalars at block level).
    
    The block-level Strassen has the same 7 terms as the 2x2x2 Strassen.
    """
    s2 = _strassen_2x2x2()
    # s2 is list of 7 tuples (u4, v4, w4) where each is length-4 vector for 2x2 indices.
    # For block level, we interpret these as coefficients for the 4 blocks.
    
    U = [[0] * 49 for _ in range(16)]  # 4x4 = 16 entries
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    col = 0
    for r in range(7):  # block-level Strassen term
        ub, vb, wb = s2[r]  # each is length-4: for blocks (00,01,10,11)
        # ub[ib*2+jb] where ib,jb are block row/col indices
        for s in range(7):  # intra-block Strassen term
            ui, vi, wi = s2[s]  # each is length-4: for intra-block indices
            # ui[i2*2+j2]
            for ib in range(2):
                for jb in range(2):
                    u_block_coeff = ub[ib * 2 + jb]
                    if u_block_coeff == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u_intra_coeff = ui[i2 * 2 + j2]
                            if u_intra_coeff == 0:
                                continue
                            i = ib * 2 + i2
                            j = jb * 2 + j2
                            U[i * 4 + j][col] += u_block_coeff * u_intra_coeff
            
            for jb in range(2):
                for kb in range(2):
                    v_block_coeff = vb[jb * 2 + kb]
                    if v_block_coeff == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v_intra_coeff = vi[j2 * 2 + k2]
                            if v_intra_coeff == 0:
                                continue
                            j = jb * 2 + j2
                            k = kb * 2 + k2
                            V[j * 4 + k][col] += v_block_coeff * v_intra_coeff
            
            for kb in range(2):
                for ib in range(2):
                    w_block_coeff = wb[kb * 2 + ib]
                    if w_block_coeff == 0:
                        continue
                    for k2 in range(2):
                        for i2 in range(2):
                            w_intra_coeff = wi[k2 * 2 + i2]
                            if w_intra_coeff == 0:
                                continue
                            k = kb * 2 + k2
                            i = ib * 2 + i2
                            W[k * 4 + i][col] += w_block_coeff * w_intra_coeff
            
            col += 1
    
    return U, V, W

def _build_schoolbook_4x4x2():
    """
    Schoolbook decomposition for 4x4 * 4x2 -> 4x2.
    Rank = 4*4*2 = 32.
    U: (16, 32), V: (8, 32), W: (8, 32)
    U[i*4+j, c], V[j*2+k, c], W[k*4+i, c]
    where i in {0..3}, j in {0..3}, k in {0..1}, c in {0..31}
    c = (i*4 + j)*2 + k
    """
    n, m, p = 4, 4, 2
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

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Build 4x4x4 Strassen part (rank 49)
    U4, V4, W4 = _build_strassen_4x4x4()
    
    # Build 4x4x2 schoolbook part (rank 32)
    U2, V2, W2 = _build_schoolbook_4x4x2()
    
    rank = 49 + 32
    
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(24)]  # m*p = 4*6 = 24
    W = [[0] * rank for _ in range(24)]  # p*n = 6*4 = 24
    
    # Copy Strassen part: handles columns k=0,1,2,3 of B
    for i in range(16):
        for c in range(49):
            U[i][c] = U4[i][c]
    
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j * 6 + k][c] = V4[j * 4 + k][c]
    
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[k * 4 + i][c] = W4[k * 4 + i][c]
    
    # Copy schoolbook part: handles columns k=4,5 of B
    offset = 49
    for i in range(16):
        for c in range(32):
            U[i][offset + c] = U2[i][c]
    
    for j in range(4):
        for k in range(2):
            for c in range(32):
                # k in schoolbook is 0,1; actual k is 4+k
                V[j * 6 + (4 + k)][offset + c] = V2[j * 2 + k][c]
    
    for k in range(2):
        for i in range(4):
            for c in range(32):
                # k in schoolbook is 0,1; actual k is 4+k
                W[(4 + k) * 4 + i][offset + c] = W2[k * 4 + i][c]
    
    return {"U": U, "V": V, "W": W}
