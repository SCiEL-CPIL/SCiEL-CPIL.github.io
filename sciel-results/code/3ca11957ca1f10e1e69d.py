"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use recursive Strassen to get rank 49 for 4x4x4
    # Strassen for 2x2: rank 7
    # For 4x4: partition into 2x2 blocks, apply Strassen at top level (7 terms),
    # each term is a 2x2 matrix multiplication, which we do with Strassen (rank 7 each)
    # Total rank = 7 * 7 = 49
    
    def strassen_2x2(U, V, W):
        """
        U: 4xR matrix (for 2x2 input, rows indexed by (i,j) with i,j in {0,1})
        V: 4xR matrix (for 2x2 input, rows indexed by (j,k) with j,k in {0,1})
        W: 4xR matrix (for 2x2 output, rows indexed by (k,i) with k,i in {0,1})
        Returns updated U, V, W with rank increased by 7
        """
        # Strassen's 7 products for 2x2 matrix multiplication
        # C = A*B where A, B are 2x2
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Current rank offset
        r0 = U.shape[1]
        
        # We need to add 7 new columns
        # For each Strassen product, we add one column to U, V, W
        
        # The 2x2 block indices:
        # A: rows 0,1 = A00, A01; rows 2,3 = A10, A11
        # B: rows 0,1 = B00, B01; rows 2,3 = B10, B11
        # C: rows 0,1 = C00, C01; rows 2,3 = C10, C11
        # (using row-major: A[i*2+j], B[j*2+k], C[k*2+i])
        
        # But wait - in our tensor decomposition, U has shape (n*m, rank) = (4, rank)
        # for 2x2, n=m=2, so U has 4 rows. V has (m*p, rank) = (4, rank). W has (p*n, rank) = (4, rank).
        # Row indexing for U: i*m+j = i*2+j for i,j in {0,1}
        # Row indexing for V: j*p+k = j*2+k for j,k in {0,1}
        # Row indexing for W: k*n+i = k*2+i for k,i in {0,1}
        
        # Strassen products - each is a bilinear form
        # We'll add 7 new columns
        
        # Product 1: (A00+A11) * (B00+B11)
        # U rows: A00 (row 0), A11 (row 3)
        # V rows: B00 (row 0), B11 (row 3)
        # W rows: contributes to C00 (row 0), C11 (row 3)
        col = r0
        U[0][col] = 1
        U[3][col] = 1
        V[0][col] = 1
        V[3][col] = 1
        W[0][col] = 1   # C00 += M1
        W[3][col] = 1   # C11 += M1
        col += 1
        
        # Product 2: (A10+A11) * B00
        # U rows: A10 (row 2), A11 (row 3)
        # V rows: B00 (row 0)
        # W rows: C10 (row 2), C11 (row 3)
        U[2][col] = 1
        U[3][col] = 1
        V[0][col] = 1
        W[2][col] = 1   # C10 += M2
        W[3][col] = -1  # C11 -= M2
        col += 1
        
        # Product 3: A00 * (B01-B11)
        # U rows: A00 (row 0)
        # V rows: B01 (row 1), B11 (row 3)
        # W rows: C01 (row 1), C11 (row 3)
        U[0][col] = 1
        V[1][col] = 1
        V[3][col] = -1
        W[1][col] = 1   # C01 += M3
        W[3][col] = 1   # C11 += M3
        col += 1
        
        # Product 4: A11 * (B10-B00)
        # U rows: A11 (row 3)
        # V rows: B10 (row 2), B00 (row 0)
        # W rows: C00 (row 0), C10 (row 2)
        U[3][col] = 1
        V[2][col] = 1
        V[0][col] = -1
        W[0][col] = 1   # C00 += M4
        W[2][col] = 1   # C10 += M4
        col += 1
        
        # Product 5: (A00+A01) * B11
        # U rows: A00 (row 0), A01 (row 1)
        # V rows: B11 (row 3)
        # W rows: C00 (row 0), C01 (row 1)
        U[0][col] = 1
        U[1][col] = 1
        V[3][col] = 1
        W[0][col] = -1  # C00 -= M5
        W[1][col] = 1   # C01 += M5
        col += 1
        
        # Product 6: (A10-A00) * (B00+B01)
        # U rows: A10 (row 2), A00 (row 0)
        # V rows: B00 (row 0), B01 (row 1)
        # W rows: C11 (row 3)
        U[2][col] = 1
        U[0][col] = -1
        V[0][col] = 1
        V[1][col] = 1
        W[3][col] = 1   # C11 += M6
        col += 1
        
        # Product 7: (A01-A11) * (B10+B11)
        # U rows: A01 (row 1), A11 (row 3)
        # V rows: B10 (row 2), B11 (row 3)
        # W rows: C00 (row 0), C11 (row 3)
        U[1][col] = 1
        U[3][col] = -1
        V[2][col] = 1
        V[3][col] = 1
        W[0][col] = 1   # C00 += M7
        W[3][col] = 1   # C11 += M7
        
        return U, V, W, r0 + 7
    
    # For 4x4, we partition into 2x2 blocks of size 2x2
    # A is 4x4, partitioned as:
    # A = [A00 A01]
    #     [A10 A11]
    # where each Aij is 2x2
    # Similarly for B and C.
    # 
    # C = A*B gives:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # We can apply Strassen at the "block" level, where each block is 2x2.
    # But each block multiplication (2x2 * 2x2) needs rank 7 (Strassen) or 8 (schoolbook).
    # If we use Strassen for both levels: 7 * 7 = 49.
    #
    # The block-level Strassen:
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B01-B11)
    # M4 = A11 * (B10-B00)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B01)
    # M7 = (A01-A11) * (B10+B11)
    #
    # Each of these is a sum/difference of 2x2 matrix multiplications.
    # For example, M1 = (A00+A11)*(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # This requires 4 separate 2x2 multiplications, each of rank 7, so M1 alone costs 28.
    # That's not efficient.
    #
    # Better approach: directly enumerate all the scalar-level Strassen decompositions.
    # For 4x4, we can use the fact that the 4x4x4 tensor has a structure that allows
    # a rank-49 decomposition via two levels of Strassen.
    #
    # Let me think of it differently. The 4x4 matrix multiplication tensor can be
    # decomposed using the Kronecker product of 2x2 Strassen with itself.
    # The 2x2 Strassen has rank 7. The Kronecker product of two rank-7 tensors has rank 49.
    # This corresponds to: treat the 4x4 multiplication as a 2x2 block multiplication
    # where each block is 2x2, and use Strassen for both the block structure and the
    # inner 2x2 multiplications.
    #
    # Implementation: We'll generate the decomposition by applying Strassen at the
    # outer level (7 terms), and for each term, which is a linear combination of
    # 2x2 matrix products, we apply Strassen at the inner level.
    #
    # Actually, the cleanest way is to just hardcode or generate the rank-49 decomposition
    # by considering all 4x4x4 = 64 scalar multiplications and replacing groups of them
    # with Strassen's 7-product scheme at both levels.
    #
    # Let me use a different approach: generate the schoolbook decomposition and then
    # apply Strassen compression at both the 2x2 block level and the scalar level.
    #
    # For now, let me just generate a valid rank-49 decomposition using the following
    # strategy: partition the 4x4 into four 2x2 blocks, and for each of the 4 block
    # products in C = A*B, use Strassen (rank 7) for the 2x2 multiplication.
    # But C00 = A00*B00 + A01*B10, which is a sum of two 2x2 products.
    # Each 2x2 product costs rank 7, so C00 costs 14, and similarly for other blocks.
    # Total: 4 * 14 = 56. That's worse than 49.
    #
    # The rank-49 comes from applying Strassen at the block level too. Let me think...
    #
    # The 4x4x4 tensor T satisfies T = T_2x2 ⊗ T_2x2 where T_2x2 is the 2x2x2 tensor.
    # The rank of T_2x2 is 7 (Strassen). The rank of T_2x2 ⊗ T_2x2 is at most 7*7 = 49.
    # This is achieved by the Kronecker product of the rank-7 decompositions.
    #
    # So I'll generate the rank-7 Strassen decomposition for 2x2, then take the
    # Kronecker product to get rank 49 for 4x4.
    
    # First, generate the rank-7 Strassen decomposition for 2x2x2
    # U_2: 4x7, V_2: 4x7, W_2: 4x7
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Strassen 2x2:
    # M1 = (A00+A11)*(B00+B11): U rows 0,3; V rows 0,3; W rows 0,3 (C00,C11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2 = (A10+A11)*B00: U rows 2,3; V row 0; W rows 2,3 (C10, -C11)
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    
    # M3 = A00*(B01-B11): U row 0; V rows 1,3; W rows 1,3 (C01, C11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # M4 = A11*(B10-B00): U row 3; V rows 2,0; W rows 0,2 (C00, C10)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # M5 = (A00+A01)*B11: U rows 0,1; V row 3; W rows 0,1 (-C00, C01)
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # M6 = (A10-A00)*(B00+B01): U rows 2,0; V rows 0,1; W row 3 (C11)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # M7 = (A01-A11)*(B10+B11): U rows 1,3; V rows 2,3; W rows 0,3 (C00, C11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1; W2[3][6] = 1
    
    # Now take the Kronecker product to get 4x4x4 rank-49 decomposition
    # The 4x4 matrix A can be seen as a 2x2 block matrix with 2x2 blocks.
    # A[i,j] where i = i_block*2 + i_local, j = j_block*2 + j_local
    # Similarly for B and C.
    #
    # U has shape (n*m, rank) = (16, 49) for 4x4
    # Row index: i*m + j = i*4 + j
    # V has shape (m*p, rank) = (16, 49)
    # Row index: j*p + k = j*4 + k
    # W has shape (p*n, rank) = (16, 49)
    # Row index: k*n + i = k*4 + i
    #
    # For the Kronecker product, each rank-1 term in the 2x2 decomposition
    # (U2[i2][c2], V2[j2][c2], W2[k2][c2]) gets expanded into a 4x4 term
    # where the local indices within each 2x2 block are also multiplied.
    #
    # The full 4x4x4 tensor is the Kronecker product of two 2x2x2 tensors.
    # A 4x4 matrix A has entries A[i,j] = A[ib*2+il, jb*2+jl]
    # where ib,jb are block indices and il,jl are local indices.
    #
    # The multiplication C = A*B in block form:
    # C[ib_cb, jb_cb] = sum over block indices
    #
    # For the Kronecker product decomposition:
    # Each rank-1 term (u2, v2, w2) in the 2x2 decomposition corresponds to
    # a set of 4x4 rank-1 terms. Specifically, if the 2x2 term has:
    #   u2 indexed by (i2, j2) where i2,j2 in {0,1}
    #   v2 indexed by (j2, k2) where j2,k2 in {0,1}
    #   w2 indexed by (k2, i2) where k2,i2 in {0,1}
    # Then in the 4x4 case, this expands to terms for each combination of
    # local indices.
    #
    # Actually, the Kronecker product of two rank-r tensors has rank at most r^2.
    # The 4x4x4 tensor T4 = T2 ⊗ T2 where T2 is the 2x2x2 multiplication tensor.
    # A rank-1 decomposition of T2 is (u, v, w) with u in R^{2x2}, v in R^{2x2}, w in R^{2x2}.
    # A rank-1 decomposition of T4 = T2⊗T2 is (u⊗u', v⊗v', w⊗w') where
    # (u,v,w) and (u',v',w') are rank-1 decompositions of T2.
    #
    # So if T2 = sum_{c=1}^{7} u_c ⊗ v_c ⊗ w_c (rank 7),
    # then T4 = sum_{c=1}^{7} sum_{c'=1}^{7} (u_c⊗u_{c'}) ⊗ (v_c⊗v_{c'}) ⊗ (w_c⊗w_{c'})
    # which has rank 49.
    #
    # Now I need to map the 2x2 indices to 4x4 indices.
    # In the 2x2 case:
    #   U rows: i*2+j for i,j in {0,1} -> indices 0,1,2,3
    #   V rows: j*2+k for j,k in {0,1} -> indices 0,1,2,3
    #   W rows: k*2+i for k,i in {0,1} -> indices 0,1,2,3
    #
    # In the 4x4 case:
    #   U rows: i*4+j for i,j in {0,1,2,3} -> indices 0..15
    #   V rows: j*4+k for j,k in {0,1,2,3} -> indices 0..15
    #   W rows: k*4+i for k,i in {0,1,2,3} -> indices 0..15
    #
    # The Kronecker product: i = ib*2+il, j = jb*2+jl, etc.
    # U row in 4x4: i*4+j = (ib*2+il)*4 + (jb*2+jl) = ib*8 + il*4 + jb*2 + jl
    # V row in 4x4: j*4+k = (jb*2+jl)*4 + (kb*2+kl) = jb*8 + jl*4 + kb*2 + kl
    # W row in 4x4: k*4+i = (kb*2+kl)*4 + (ib*2+il) = kb*8 + kl*4 + ib*2 + il
    #
    # For the Kronecker term (c, c'):
    # U_4[i*4+j][c*7+c'] = U2[ib*2+jb][c] * U2[il*2+jl][c']
    # V_4[j*4+k][c*7+c'] = V2[jb*2+kb][c] * V2[jl*2+kl][c']
    # W_4[k*4+i][c*7+c'] = W2[kb*2+ib][c] * W2[kl*2+il][c']
    
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for c in range(7):
        for cp in range(7):
            col = c * 7 + cp
            # U: for each (ib, jb, il, jl)
            for ib in range(2):
                for jb in range(2):
                    for il in range(2):
                        for jl in range(2):
                            i = ib * 2 + il
                            j = jb * 2 + jl
                            u_val = U2[ib*2+jb][c] * U2[il*2+jl][cp]
                            if u_val != 0:
                                U[i*4+j][col] = u_val
            # V: for each (jb, kb, jl, kl)
            for jb in range(2):
                for kb in range(2):
                    for jl in range(2):
                        for kl in range(2):
                            j = jb * 2 + jl
                            k = kb * 2 + kl
                            v_val = V2[jb*2+kb][c] * V2[jl*2+kl][cp]
                            if v_val != 0:
                                V[j*4+k][col] = v_val
            # W: for each (kb, ib, kl, il)
            for kb in range(2):
                for ib in range(2):
                    for kl in range(2):
                        for il in range(2):
                            k = kb * 2 + kl
                            i = ib * 2 + il
                            w_val = W2[kb*2+ib][c] * W2[kl*2+il][cp]
                            if w_val != 0:
                                W[k*4+i][col] = w_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
