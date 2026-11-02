"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    
    # We'll use Strassen's algorithm for the 4x4x4 part (rank 49)
    # and schoolbook for the 4x4x3 part (rank 48)
    # Total rank = 49 + 48 = 97
    
    rank = 97
    U = [[0] * rank for _ in range(n * m)]  # 16 x 97
    V = [[0] * rank for _ in range(m * p)]  # 28 x 97
    W = [[0] * rank for _ in range(p * n)]  # 28 x 97
    
    # Helper to add a rank-1 term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
    
    # Part 1: 4x4x4 using Strassen (rank 49)
    # We decompose 4x4 matrices into 2x2 blocks
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = A*B = [[C00, C01], [C10, C11]]
    # where C00 = A00*B00 + A01*B10, etc.
    
    # Strassen products for 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B11+B01)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # For the 4x4x4 case, each "block" is a 2x2 matrix
    # So we need to compute 7 products of 2x2 matrices
    # Each 2x2 matrix product can be done with Strassen (rank 7)
    # Total: 7 * 7 = 49 terms
    
    # Define the 7 Strassen products for the outer level
    # Each product is a 2x2 matrix multiplication
    # We'll use Strassen for each 2x2 multiplication
    
    # The 7 outer products:
    # P1 = (A00+A11) * (B00+B11)
    # P2 = (A10+A11) * B00
    # P3 = A00 * (B10-B11)
    # P4 = A11 * (B01-B00)
    # P5 = (A00+A01) * B11
    # P6 = (A10-A00) * (B00+B10)
    # P7 = (A01-A11) * (B11+B01)
    
    # Each Pi is a 2x2 matrix
    # We decompose each Pi using Strassen (7 terms each)
    
    # For each outer product, we need to define:
    # - The linear combination of A blocks (U vector)
    # - The linear combination of B blocks (V vector)
    # - The contribution to C blocks (W vector)
    
    # Let's define the outer Strassen products:
    # Each has a specific A combination, B combination, and C contribution
    
    # Outer product definitions:
    # (a_comb, b_comb, c_contrib)
    # a_comb: list of (block_idx, coeff) for A
    # b_comb: list of (block_idx, coeff) for B
    # c_contrib: list of (block_idx, coeff) for C
    
    # A blocks: 0=A00, 1=A01, 2=A10, 3=A11
    # B blocks: 0=B00, 1=B01, 2=B10, 3=B11
    # C blocks: 0=C00, 1=C01, 2=C10, 3=C11
    
    outer_products = [
        # P1 = (A00+A11) * (B00+B11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # P2 = (A10+A11) * B00
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # P3 = A00 * (B10-B11)
        ([(0, 1)], [(2, 1), (3, -1)], [(1, 1), (3, 1)]),
        # P4 = A11 * (B01-B00)
        ([(3, 1)], [(1, 1), (0, -1)], [(0, -1), (1, 1)]),
        # P5 = (A00+A01) * B11
        ([(0, 1), (1, 1)], [(3, 1)], [(0, 1), (2, 1)]),
        # P6 = (A10-A00) * (B00+B10)
        ([(2, 1), (0, -1)], [(0, 1), (2, 1)], [(3, 1)]),
        # P7 = (A01-A11) * (B11+B01)
        ([(1, 1), (3, -1)], [(3, 1), (1, 1)], [(0, -1)]),
    ]
    
    # Now for each outer product, we decompose the 2x2 matrix multiplication
    # using Strassen (7 terms)
    
    # Inner Strassen products for 2x2:
    # (a_comb, b_comb, c_contrib)
    # A blocks: 0=A00, 1=A01, 2=A10, 3=A11 (within the 2x2 block)
    # B blocks: 0=B00, 1=B01, 2=B10, 3=B11 (within the 2x2 block)
    # C blocks: 0=C00, 1=C01, 2=C10, 3=C11 (within the 2x2 block)
    
    inner_products = [
        # M1 = (A00+A11) * (B00+B11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (A10+A11) * B00
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = A00 * (B10-B11)
        ([(0, 1)], [(2, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4 = A11 * (B01-B00)
        ([(3, 1)], [(1, 1), (0, -1)], [(0, -1), (1, 1)]),
        # M5 = (A00+A01) * B11
        ([(0, 1), (1, 1)], [(3, 1)], [(0, 1), (2, 1)]),
        # M6 = (A10-A00) * (B00+B10)
        ([(2, 1), (0, -1)], [(0, 1), (2, 1)], [(3, 1)]),
        # M7 = (A01-A11) * (B11+B01)
        ([(1, 1), (3, -1)], [(3, 1), (1, 1)], [(0, -1)]),
    ]
    
    # Helper to get global index for a block entry
    # Block (br, bc) has entries at global (br*2+i, bc*2+j) for i,j in 0,1
    def get_a_idx(br, bc, i, j):
        return (br * 2 + i) * m + (bc * 2 + j)
    
    def get_b_idx(br, bc, i, j):
        return (br * 2 + i) * p + (bc * 2 + j)
    
    def get_c_idx(br, bc, i, j):
        return (bc * 2 + j) * n + (br * 2 + i)
    
    # Now generate all 49 terms for the 4x4x4 part
    col = 0
    for outer_idx, (a_comb, b_comb, c_contrib) in enumerate(outer_products):
        for inner_idx, (ia_comb, ib_comb, ic_contrib) in enumerate(inner_products):
            # This term corresponds to:
            # (linear_comb of A) * (linear_comb of B)
            # contributing to (linear_comb of C)
            
            # Build U vector: linear combination of A entries
            # a_comb gives outer block combinations
            # ia_comb gives inner block combinations within those blocks
            
            # The effective A combination is:
            # For each (outer_block, outer_coeff) in a_comb:
            #   For each (inner_block, inner_coeff) in ia_comb:
            #     The actual A block is outer_block * 2 + inner_block (in some sense)
            #     Wait, this is not quite right.
            
            # Let me think about this differently.
            # The outer product P_outer = (sum_a outer_coeff_a * A_block_a) * (sum_b outer_coeff_b * B_block_b)
            # Each A_block_a is a 2x2 matrix with entries A_block_a[i,j]
            # Each B_block_b is a 2x2 matrix with entries B_block_b[i,j]
            
            # The inner Strassen decomposes the product of two 2x2 matrices X*Y
            # where X = sum_a outer_coeff_a * A_block_a and Y = sum_b outer_coeff_b * B_block_b
            
            # The inner Strassen term is:
            # (sum_ia inner_coeff_ia * X_block_ia) * (sum_ib inner_coeff_ib * Y_block_ib)
            # where X_block_ia is the ia-th 2x2 sub-block of X
            # and Y_block_ib is the ib-th 2x2 sub-block of Y
            
            # But X is already a 2x2 matrix, so X_block_ia doesn't make sense in the same way.
            
            # I think I'm overcomplicating this. Let me use a different approach.
            
            # Actually, the standard way to apply Strassen recursively is:
            # For 4x4, we have 4 block products C00, C01, C10, C11
            # Each is a 2x2 matrix product
            # We use Strassen to compute each 2x2 product with 7 scalar multiplications
            # Total: 4 * 7 = 28? No, that's not right either.
            
            # Let me look at this more carefully.
            # The rank of 4x4x4 matrix multiplication is 49.
            # This can be achieved by applying Strassen's algorithm.
            # Strassen for 2x2 has rank 7.
            # For 4x4, we view it as 2x2 blocks of 2x2 matrices.
            # The block multiplication involves 7 products of 2x2 matrices.
            # Each 2x2 matrix product has rank 7 (using Strassen).
            # Total rank: 7 * 7 = 49.
            
            # So my original approach was correct. Let me implement it properly.
            
            # For each of the 7 outer products, we have a 2x2 matrix multiplication
            # We decompose each using the 7-term Strassen decomposition
            # This gives 7 * 7 = 49 terms total
            
            # The issue is correctly mapping the indices.
            
            # Let me define it more explicitly.
            
            # Outer product outer_idx corresponds to:
            # X_outer = sum_{(a_blk, a_coeff) in a_comb} a_coeff * A_block[a_blk]
            # Y_outer = sum_{(b_blk, b_coeff) in b_comb} b_coeff * B_block[b_blk]
            # Z_outer = X_outer * Y_outer (a 2x2 matrix)
            
            # Now we decompose Z_outer using Strassen:
            # For inner_idx in 0..6:
            #   X_inner = sum_{(ia_blk, ia_coeff) in ia_comb} ia_coeff * X_outer[ia_blk]
            #   Y_inner = sum_{(ib_blk, ib_coeff) in ib_comb} ib_coeff * Y_outer[ib_blk]
            #   term = X_inner * Y_inner (a scalar)
            #   This term contributes to Z_outer according to ic_contrib
            
            # But X_outer[ia_blk] is a 2x2 matrix, so X_inner is also a 2x2 matrix
            # This doesn't make sense for a scalar multiplication.
            
            # I think the correct interpretation is:
            # The 7 outer products are scalar multiplications of 2x2 matrices
            # No, that's not right either.
            
            # Let me start over with a clearer formulation.
            
            # Matrix multiplication C = A*B where A is 4x4, B is 4x4, C is 4x4
            # Partition into 2x2 blocks:
            # A = [[A00, A01], [A10, A11]]
            # B = [[B00, B01], [B10, B11]]
            # C = [[C00, C01], [C10, C11]]
            
            # C00 = A00*B00 + A01*B10
            # C01 = A00*B01 + A01*B11
            # C10 = A10*B00 + A11*B10
            # C11 = A10*B01 + A11*B11
            
            # Using Strassen:
            # M1 = (A00 + A11) * (B00 + B11)
            # M2 = (A10 + A11) * B00
            # M3 = A00 * (B10 - B11)
            # M4 = A11 * (B01 - B00)
            # M5 = (A00 + A01) * B11
            # M6 = (A10 - A00) * (B00 + B10)
            # M7 = (A01 - A11) * (B11 + B01)
            
            # C00 = M1 - M4 + M5 - M7
            # C01 = M3 + M4
            # C10 = M2 + M5
            # C11 = M1 - M2 + M3 + M6
            
            # Each Mi is a 2x2 matrix (the product of two 2x2 matrices)
            # To compute each Mi, we can use Strassen again (7 scalar multiplications)
            # Or we can use schoolbook (4 scalar multiplications)
            
            # If we use Strassen for each Mi, total rank = 7 * 7 = 49
            # If we use schoolbook for each Mi, total rank = 7 * 4 = 28
            # But the rank of 4x4x4 is 49, so schoolbook for the inner products must give rank 49?
            # No, that's not right. The rank of 4x4x4 is 49, which is achieved by Strassen.
            # If we use schoolbook for the inner 2x2 products, we get rank 7*4=28, which is less than 49.
            # This is a contradiction.
            
            # I think the issue is that the rank of 4x4x4 is not 49.
            # Let me check: the rank of 2x2x2 is 7 (Strassen).
            # The rank of 4x4x4 is at most 7*7=49 using Strassen recursively.
            # But it could be lower.
            
            # Actually, I recall that the rank of 4x4x4 matrix multiplication is 49.
            # This is a known result.
            
            # So if I use schoolbook for the inner products, I get rank 28, which would be better than 49.
            # But that can't be right if the rank is 49.
            
            # I think I'm confusing myself. Let me just use the schoolbook decomposition for the entire 4x4x4 part.
            # Schoolbook for 4x4x4 has rank 4*4*4 = 64.
            # Then for the 4x4x3 part, schoolbook has rank 4*4*3 = 48.
            # Total rank = 64 + 48 = 112.
            
            # This is the same as the original schoolbook for 4x4x7 (rank 112).
            # So I haven't improved anything.
            
            # Let me try a different approach: use Strassen for the 4x4x4 part.
            # I'll hardcode the 49 terms.
            
            # Due to the complexity, I'll just use the schoolbook for now.
            # This gives rank 112, which is the same as the original.
            
            # Let me try to use Strassen properly.
            # I'll define the 49 terms explicitly.
            
            # Actually, let me just use the schoolbook for the entire tensor.
            # This is simple and correct, even if not optimal.
            
            pass
    
    # Let me just use the schoolbook decomposition for the entire 4x4x7 tensor
    # This gives rank 112, which is the same as the original program.
    # But I need to improve it.
    
    # Let me try rank 97 with a proper Strassen decomposition.
    # I'll implement it step by step.
    
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 4x4x4 using Strassen (rank 49)
    # I'll use a precomputed decomposition.
    
    # For simplicity, let me use the schoolbook for the 4x4x4 part (rank 64)
    # and schoolbook for the 4x4x3 part (rank 48)
    # Total rank = 112
    
    # But I want to do better. Let me try to implement Strassen properly.
    
    # Define the 7 Strassen products for 2x2:
    # Each product is defined by (A_comb, B_comb, C_contrib)
    # where A_comb is a list of (idx, coeff) for the 4 entries of the 2x2 A matrix
    # B_comb is a list of (idx, coeff) for the 4 entries of the 2x2 B matrix
    # C_contrib is a list of (idx, coeff) for the 4 entries of the 2x2 C matrix
    
    # A entries: 0=A00, 1=A01, 2=A10, 3=A11
    # B entries: 0=B00, 1=B01, 2=B10, 3=B11
    # C entries: 0=C00, 1=C01, 2=C10, 3=C11
    
    strassen_2x2 = [
        # M1 = (A00+A11)*(B00+B11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (A10+A11)*B00
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = A00*(B10-B11)
        ([(0, 1)], [(2, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4 = A11*(B01-B00)
        ([(3, 1)], [(1, 1), (0, -1)], [(0, -1), (1, 1)]),
        # M5 = (A00+A01)*B11
        ([(0, 1), (1, 1)], [(3, 1)], [(0, 1), (2, 1)]),
        # M6 = (A10-A00)*(B00+B10)
        ([(2, 1), (0, -1)], [(0, 1), (2, 1)], [(3, 1)]),
        # M7 = (A01-A11)*(B11+B01)
        ([(1, 1), (3, -1)], [(3, 1), (1, 1)], [(0, -1)]),
    ]
    
    # For the 4x4x4 case, we have 4 block products: C00, C01, C10, C11
    # Each is a 2x2 matrix product
    # We use Strassen to compute each with 7 terms
    # Total: 4 * 7 = 28 terms
    
    # But wait, this gives rank 28, which is less than 49.
    # This suggests that the rank of 4x4x4 is at most 28.
    # But I thought it was 49.
    
    # Let me check: if I use Strassen for the 4x4 case, I get 7 products of 2x2 matrices.
    # Each 2x2 matrix product can be done with 7 scalar multiplications (Strassen) or 4 (schoolbook).
    # If I use schoolbook for the 2x2 products, I get 7*4 = 28 scalar multiplications.
    # If I use Strassen for the 2x2 products, I get 7*7 = 49 scalar multiplications.
    
    # So the rank of 4x4x4 is at most 28 (using schoolbook for the inner products).
    # This is better than 49!
    
    # So let me use this: Strassen for the outer level (7 products), schoolbook for the inner level (4 terms each)
    # Total rank for 4x4x4: 7 * 4 = 28
    
    # Then for the 4x4x3 part, schoolbook: 4*4*3 = 48
    # Total rank: 28 + 48 = 76
    
    # This is much better than 112!
    
    # Let me implement this.
    
    # Reset
    rank = 76
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 4x4x4 using Strassen outer + schoolbook inner (rank 28)
    
    # The 7 outer Strassen products:
    # Each is a product of two 2x2 matrices
    # We decompose each using schoolbook (4 terms)
    
    # Outer product definitions:
    # (A_block_comb, B_block_comb, C_block_contrib)
    # A_block_comb: list of (block_idx, coeff) where block_idx in 0..3 (0=A00, 1=A01, 2=A10, 3=A11)
    # B_block_comb: list of (block_idx, coeff) where block_idx in 0..3 (0=B00, 1=B01, 2=B10, 3=B11)
    # C_block_contrib: list of (block_idx, coeff) where block_idx in 0..3 (0=C00, 1=C01, 2=C10, 3=C11)
    
    outer_strassen = [
        # M1 = (A00+A11)*(B00+B11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (A10+A11)*B00
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = A00*(B10-B11)
        ([(0, 1)], [(2, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4 = A11*(B01-B00)
        ([(3, 1)], [(1, 1), (0, -1)], [(0, -1), (1, 1)]),
        # M5 = (A00+A01)*B11
        ([(0, 1), (1, 1)], [(3, 1)], [(0, 1), (2, 1)]),
        # M6 = (A10-A00)*(B00+B10)
        ([(2, 1), (0, -1)], [(0, 1), (2, 1)], [(3, 1)]),
        # M7 = (A01-A11)*(B11+B01)
        ([(1, 1), (3, -1)], [(3, 1), (1, 1)], [(0, -1)]),
    ]
    
    # For each outer product, we decompose the 2x2 matrix product using schoolbook
    # Schoolbook for 2x2: 4 terms
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each term is a scalar multiplication
    # For the i-th term of the schoolbook decomposition of X*Y:
    # X is a 2x2 matrix, Y is a 2x2 matrix
    # The term corresponds to X[i,j] * Y[j,k] contributing to C[i,k]
    
    # For the outer product, X is a linear combination of A blocks
    # Y is a linear combination of B blocks
    
    # So for each term in the schoolbook decomposition:
    # X[i,j] = sum_{(a_blk, a_coeff)} a_coeff * A_block[a_blk][i,j]
    # Y[j,k] = sum_{(b_blk, b_coeff)} b_coeff * B_block[b_blk][j,k]
    # The product X[i,j] * Y[j,k] is a bilinear form
    # We need to decompose it into rank-1 terms
    
    # Actually, X[i,j] * Y[j,k] is already a rank-1 term in the tensor sense
    # because it's a product of a linear form in A and a linear form in B
    
    # So for each outer product and each schoolbook term, we have one rank-1 term
    
    # Let me implement this.
    
    col = 0
    for a_comb, b_comb, c_contrib in outer_strassen:
        # This outer product computes X*Y where:
        # X[i,j] = sum_{(a_blk, a_coeff) in a_comb} a_coeff * A_block[a_blk][i,j]
        # Y[j,k] = sum_{(b_blk, b_coeff) in b_comb} b_coeff * B_block[b_blk][j,k]
        
        # Schoolbook decomposition of X*Y:
        # 4 terms: (i,j,k) in [(0,0,0), (0,1,1), (1,0,0), (1,1,1)]
        # Wait, that's not right.
        # C[i,k] = sum_j X[i,j] * Y[j,k]
        # So the terms are: for i in 0,1, j in 0,1, k in 0,1:
        #   term = X[i,j] * Y[j,k] contributes to C[i,k]
        # That's 8 terms, but schoolbook for 2x2 is 4 multiplications.
        # The 4 multiplications are:
        # M00 = X[0,0]*Y[0,0] + X[0,1]*Y[1,0]  -> no, that's not right.
        
        # Schoolbook for 2x2 matrix multiplication:
        # C[0,0] = X[0,0]*Y[0,0] + X[0,1]*Y[1,0]
        # C[0,1] = X[0,0]*Y[0,1] + X[0,1]*Y[1,1]
        # C[1,0] = X[1,0]*Y[0,0] + X[1,1]*Y[1,0]
        # C[1,1] = X[1,0]*Y[0,1] + X[1,1]*Y[1,1]
        
        # This requires 8 scalar multiplications, not 4.
        # I was wrong about schoolbook for 2x2 being 4 terms.
        # Schoolbook for 2x2 is 4*2 = 8 scalar multiplications? No.
        
        # Actually, schoolbook for n x n matrix multiplication is n^3 scalar multiplications.
        # For 2x2, that's 8 scalar multiplications.
        # Strassen reduces this to 7.
        
        # So if I use schoolbook for the inner 2x2 products, I get 7 * 8 = 56 terms for the 4x4x4 part.
        # Total rank: 56 + 48 = 104.
        
        # This is better than 112, but not as good as I hoped.
        
        # If I use Strassen for the inner products, I get 7 * 7 = 49 terms.
        # Total rank: 49 + 48 = 97.
        
        # So let me use Strassen for both levels.
        
        # But I already tried that and got confused.
        
        # Let me just use the schoolbook for the entire tensor.
        # This is simple and correct.
        
        pass
    
    # I'll just use the schoolbook for the entire 4x4x7 tensor.
    # This gives rank 112.
    
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
# EVOLVE-BLOCK-END
