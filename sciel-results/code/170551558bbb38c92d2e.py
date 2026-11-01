"""Direct rank-28 construction for 4x4 matrix multiplication using 2x2 Strassen Kronecker embedding."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # 2x2 Strassen tensors
    # u2[7][4], v2[7][4], w2[7][4]
    # Indices: 0=a, 1=b, 2=c, 3=d (input)
    # Output indices: 0=e, 1=f, 2=g, 3=h
    u2 = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 1],
    ]
    v2 = [
        [1, 0, 0, 0],
        [1, 0, 1, 0],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
    w2 = [
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, -1],
        [1, 0, -1, 0],
        [0, 1, 0, 1],
        [0, -1, 1, 0],
    ]
    
    n, m, p = DIMENSIONS  # 4, 4, 4
    rank = 28  # 7 * 4
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
    
    for r1 in range(7):
        for r2 in range(4):
            r = r1 * 4 + r2
            for idx_u in range(16):
                i = idx_u // 4
                j = idx_u % 4
                i0 = i % 2
                j0 = j % 2
                # Only contribute when i//2 == 0 and j//2 == 0 (top-left 2x2 block)
                # But actually, we embed the 2x2 Strassen into the top-left 2x2 block of 4x4
                # So we need to handle the embedding properly
                # The Kronecker product approach: A = A11 ⊗ I + ... 
                # Let's use the standard embedding where 4x4 multiplication is decomposed
                # into 2x2 block matrix multiplication using 2x2 Strassen
                # 
                # For block matrix multiplication:
                # [A11 A12] [B11 B12] = [A11B11+A12B21  A11B12+A12B22]
                # [A21 A22] [B21 B22]   [A21B11+A22B21  A21B12+A22B22]
                #
                # Each 2x2 block multiplication uses 7 Strassen products
                # Total 28 products
                #
                # The embedding maps:
                # Block A11 -> positions (0,0),(0,1),(1,0),(1,1)
                # Block A12 -> positions (0,2),(0,3),(1,2),(1,3)
                # Block A21 -> positions (2,0),(2,1),(3,0),(3,1)
                # Block A22 -> positions (2,2),(2,3),(3,2),(3,3)
                #
                # For the Kronecker product construction, the rank-28 decomposition
                # corresponds to applying 2x2 Strassen to each of the 4 block products
                #
                # Let's build this directly using the tensor network structure
                # U[i*m+j, r] where r = r1*4 + r2
                # The factorization decomposes as:
                # C = sum_r U_r ⊗ V_r ⊗ W_r (outer product form)
                #
                # For the 4x4 case with 2x2 Strassen blocks:
                # We partition indices into (block, local) where block in {0,1} for 2x2 blocks
                # i = block_i * 2 + local_i, j = block_j * 2 + local_j
                # k = block_k * 2 + local_k
                #
                # The rank-28 indices: r1 in 0..6 (Strassen product), r2 in 0..3 (block combination)
                # r2 encodes which of the 4 block products: 0=A11B11, 1=A12B21, 2=A21B11, 3=A22B21? No...
                # Actually for block matrix mult, we have 4 block products, each using 7 Strassen factors
                # So r2 in 0..3 selects the block product
                pass
            
    # Let me rebuild this more carefully using the known rank-28 tensor
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # The 28-term decomposition for 4x4 matrix multiplication
    # derived from 2x2 Strassen applied to block structure
    # 
    # Block structure:
    # A = [A11 A12], B = [B11 B12], C = [C11 C12]
    #     [A21 A22]     [B21 B22]     [C21 C22]
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # Each 2x2 product (like A11*B11) uses 7 Strassen factors
    # Total: 4 blocks * 7 = 28 factors
    #
    # For block product X*Y where X,Y are 2x2:
    # Uses u2[r1], v2[r1], w2[r1] for r1 in 0..6
    #
    # Mapping for block (bi, bj) where bi is row block of A, bj is col block of B:
    # A block (bi, bj) has entries at positions:
    #   row: bi*2 + i_local, col: bj*2 + j_local
    # For U (input to A): U[bi*4 + bj*2 + i_local*2 + j_local, r]
    # Wait, U has size 16 x 28, indexed by [i*4+j] where i,j in 0..3
    # So for block (bi, bj), local (i_loc, j_loc):
    #   global index = (bi*2 + i_loc)*4 + (bj*2 + j_loc)
    
    for r1 in range(7):
        for block_idx in range(4):
            r = r1 * 4 + block_idx
            # block_idx: 0=C11 from A11*B11, 1=C11 from A12*B21, 2=C12 from A11*B12, 3=C12 from A12*B22
            # Actually let's map: block_idx 0->(A11,B11)->C11, 1->(A12,B21)->C11
            #                 block_idx 2->(A11,B12)->C12, 3->(A12,B22)->C12
            #                 block_idx 4->(A21,B11)->C21, ... but we only have 4 blocks
            # Wait, there are 4 output blocks, each requiring 2 block products
            # C11: A11*B11 + A12*B21 (2 products)
            # C12: A11*B12 + A12*B22 (2 products)
            # C21: A21*B11 + A22*B21 (2 products)
            # C22: A21*B12 + A22*B22 (2 products)
            # Total 8 block products * 7 = 56? No, that's not right.
            # 
            # Actually, the standard approach uses the fact that 4x4 matrix multiplication
            # can be done with 28 multiplications by viewing it as 2x2 block matrix
            # multiplication where each block is 2x2, and using Strassen for the block mult.
            # The 2x2 block matrix multiplication requires 7 block multiplications (Strassen),
            # and each block multiplication of 2x2 matrices requires 7 scalar multiplications.
            # Total: 7 * 7 = 49. That's not 28.
            #
            # Let me reconsider. The rank of 4x4 matrix multiplication is at most 49
            # (from Kronecker product of two 2x2 Strassen). To get 28, we need a different approach.
            #
            # Actually, the known result is that the rank of 4x4 matrix multiplication is at most 28
            # is NOT correct. The upper bound from Kronecker product is 49.
            # 
            # Let me just implement the rank-49 Kronecker product construction which is valid.
            pass
    
    # Implement rank-49 Kronecker product of two 2x2 Strassen decompositions
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for idx_u in range(16):
                i = idx_u // 4
                j = idx_u % 4
                i1, i0 = divmod(i, 2)
                j1, j0 = divmod(j, 2)
                val_u = u2[r1][i1*2+j1] * u2[r2][i0*2+j0]
                U[idx_u][r] = val_u
            
            for idx_v in range(16):
                j = idx_v // 4
                k = idx_v % 4
                j1, j0 = divmod(j, 2)
                k1, k0 = divmod(k, 2)
                val_v = v2[r1][j1*2+k1] * v2[r2][j0*2+k0]
                V[idx_v][r] = val_v
            
            for idx_w in range(16):
                k = idx_w // 4
                i = idx_w % 4
                k1, k0 = divmod(k, 2)
                i1, i0 = divmod(i, 2)
                val_w = w2[r1][k1*2+i1] * w2[r2][k0*2+i0]
                W[idx_w][r] = val_w
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
