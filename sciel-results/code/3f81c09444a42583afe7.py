"""Matrix multiplication 4x4 x 4x5: rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # We need a rank-minimal exact tensor decomposition for multiplying
    # a 4x4 matrix by a 4x5 matrix.
    #
    # The schoolbook decomposition has rank 4*4*5 = 80.
    #
    # For small matrices, we can use known efficient decompositions.
    # The 2x2x2 matrix multiplication has rank 7 (Laderman's algorithm).
    # We can use block-wise application of such algorithms.
    #
    # Strategy: Use a block decomposition where we partition the matrices
    # into smaller blocks and use efficient decompositions for each block.
    #
    # For 4x4 * 4x5, we can think of this as:
    # - Split the 4x4 matrix A into four 2x2 blocks
    # - Split the 4x5 matrix B into a 2x2 block of submatrices: 
    #   B = [[B11(2x2), B12(2x3)], [B21(2x2), B22(2x3)]]
    # - The result C = A*B is 4x5, split into four 2x(5/2) blocks... 
    #   Actually this is awkward with odd dimensions.
    #
    # Let me think differently. The tensor for 4x4*4x5 multiplication has
    # indices (i,j,k) where i in [0,4), j in [0,4), k in [0,5).
    # 
    # The rank of the matrix multiplication tensor M_{n,m,p} is known to be:
    # - M_{2,2,2} has rank 7
    # - M_{3,3,3} has rank at most 23 (Strassen-like), conjectured to be lower
    #
    # For our case 4,4,5, we can try:
    # 1. First use Strassen's algorithm on 2x2 blocks for the 4x4*4x4 part
    #    and handle the 5th column separately.
    # 2. Or use a direct rank-reduction approach.
    #
    # Let's use a hybrid approach:
    # - Decompose the 4x4*4x4 part using Strassen (rank 7 per 2x2x2 block)
    # - The 4x4*4x1 (for the 5th column) part uses rank 4*4*1=16 schoolbook
    #   But we can do better...
    #
    # Actually, let's think about this more carefully.
    #
    # The multiplication A(4x4) * B(4x5) = C(4x5) can be viewed as:
    # C[:,0:4] = A * B[:,0:4]  (4x4 * 4x4)
    # C[:,4] = A * B[:,4]      (4x4 * 4x1)
    #
    # For the 4x4*4x4 part, using Strassen recursively:
    # - 4x4*4x4 can be done with 7*(2x2*2x2) operations at the top level
    # - Each 2x2*2x2 uses rank 7
    # - Total rank for 4x4*4x4 = 7*7 = 49 (using Strassen twice)
    #
    # Wait, that's not quite right. Strassen's algorithm for 2x2*2x2 uses
    # 7 multiplications of 1x1 matrices. For 4x4*4x4, we group into 2x2 blocks
    # and use Strassen, giving 7 "multiplications" of 2x2 matrices, each of
    # which can be done with rank 7, so total rank = 7*7 = 49.
    #
    # But actually, the rank of M_{4,4,4} is known to be at most 49 by Strassen,
    # and it's conjectured to be lower. The current best upper bound is 49.
    #
    # For the 4x4*4x1 part, the rank is at most 4*4*1 = 16 by schoolbook.
    # Can we do better? The rank of M_{4,4,1} is the rank of a 4x4 matrix 
    # multiplication with a vector, which is essentially just 16 independent
    # dot products. Actually, M_{n,m,1} has rank n*m since each output C[i]
    # depends on all A[i,j]*B[j] terms, and these are all independent.
    #
    # So total rank would be 49 + 16 = 65.
    #
    # But wait, we can potentially share some structure. Let me think about
    # whether we can do better by not separating the two parts.
    #
    # Actually, let me just implement the Strassen-based approach for the
    # 4x4*4x4 part and schoolbook for the 4x4*4x1 part.
    #
    # First, let me implement Strassen's algorithm for 2x2*2x2 multiplication
    # as a rank-7 decomposition.
    #
    # For 2x2 matrices A, B, C:
    # C = A*B
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    #
    # Strassen's 7 products:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each M is a product of sums/differences of entries, so each M corresponds
    # to one rank-1 term in the decomposition.
    #
    # For the full 2x2x2 tensor, the rank-7 decomposition has:
    # U matrix (4x7): rows are (i,j) pairs, columns are the 7 Strassen products
    # V matrix (4x7): rows are (j,k) pairs, columns are the 7 Strassen products
    # W matrix (4x7): rows are (k,i) pairs, columns are the 7 Strassen products
    #
    # Let me construct this carefully.
    
    def strassen_2x2_decomp():
        """Return U, V, W for 2x2*2x2 matrix multiplication with rank 7."""
        # Indices: A has indices (i,j), B has (j,k), C has (k,i)
        # i, j, k in {0, 1}
        # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        
        # The 7 Strassen products and their contributions to U, V, W
        # M1 = (A11+A22)*(B11+B22)
        #   U contributions: A11=(0,0), A22=(1,1)
        #   V contributions: B11=(0,0), B22=(1,1)
        #   W contributions: This product contributes to C11 and C22
        #     C11 += M1, C22 += M1
        #     C11 is (k=0,i=0), C22 is (k=1,i=1)
        
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        # M1 = (A11+A22)*(B11+B22)
        # U: A11 at (0,0)=row 0, A22 at (1,1)=row 3
        U[0][0] = 1  # A11
        U[3][0] = 1  # A22
        # V: B11 at (0,0)=row 0, B22 at (1,1)=row 3
        V[0][0] = 1  # B11
        V[3][0] = 1  # B22
        # W: C11 at (0,0)=row 0, C22 at (1,1)=row 3
        W[0][0] = 1  # C11 = (k=0, i=0) -> index k*2+i = 0
        W[3][0] = 1  # C22 = (k=1, i=1) -> index k*2+i = 3
        
        # M2 = (A21+A22)*B11
        # U: A21 at (1,0)=row 2, A22 at (1,1)=row 3
        U[2][1] = 1
        U[3][1] = 1
        # V: B11 at (0,0)=row 0
        V[0][1] = 1
        # W: C21 += M2, C22 -= M2
        # C21 = (k=0, i=1) -> index 0*2+1 = 1
        # C22 = (k=1, i=1) -> index 3
        W[1][1] = 1
        W[3][1] = -1
        
        # M3 = A11*(B12-B22)
        # U: A11 at (0,0)=row 0
        U[0][2] = 1
        # V: B12 at (0,1)=row 1, B22 at (1,1)=row 3
        V[1][2] = 1
        V[3][2] = -1
        # W: C12 += M3, C22 += M3
        # C12 = (k=1, i=0) -> index 1*2+0 = 2
        # C22 = (k=1, i=1) -> index 3
        W[2][2] = 1
        W[3][2] = 1
        
        # M4 = A22*(B21-B11)
        # U: A22 at (1,1)=row 3
        U[3][3] = 1
        # V: B21 at (1,0)=row 2, B11 at (0,0)=row 0
        V[2][3] = 1
        V[0][3] = -1
        # W: C11 += M4, C21 += M4
        # C11 = (k=0, i=0) -> index 0
        # C21 = (k=0, i=1) -> index 1
        W[0][3] = 1
        W[1][3] = 1
        
        # M5 = (A11+A12)*B22
        # U: A11 at (0,0)=row 0, A12 at (0,1)=row 1
        U[0][4] = 1
        U[1][4] = 1
        # V: B22 at (1,1)=row 3
        V[3][4] = 1
        # W: C11 -= M5, C12 += M5
        # C11 = (k=0, i=0) -> index 0
        # C12 = (k=1, i=0) -> index 2
        W[0][4] = -1
        W[2][4] = 1
        
        # M6 = (A21-A11)*(B11+B12)
        # U: A21 at (1,0)=row 2, A11 at (0,0)=row 0
        U[2][5] = 1
        U[0][5] = -1
        # V: B11 at (0,0)=row 0, B12 at (0,1)=row 1
        V[0][5] = 1
        V[1][5] = 1
        # W: C22 += M6
        # C22 = (k=1, i=1) -> index 3
        W[3][5] = 1
        
        # M7 = (A12-A22)*(B21+B22)
        # U: A12 at (0,1)=row 1, A22 at (1,1)=row 3
        U[1][6] = 1
        U[3][6] = -1
        # V: B21 at (1,0)=row 2, B22 at (1,1)=row 3
        V[2][6] = 1
        V[3][6] = 1
        # W: C11 += M7
        # C11 = (k=0, i=0) -> index 0
        W[0][6] = 1
        
        return U, V, W
    
    def expand_decomp(base_U, base_V, base_W, n_blocks_i, n_blocks_j, block_size_i, block_size_j, block_size_k):
        """
        Expand a base decomposition from block_size to larger dimensions.
        This is for the recursive Strassen approach.
        """
        pass  # We'll handle the expansion manually
    
    # For 4x4*4x5:
    # Split into:
    # - 4x4*4x4 part (using Strassen twice, rank 49)
    # - 4x4*4x1 part (schoolbook, rank 16)
    # Total rank = 49 + 16 = 65
    
    # But let's verify: can we do better?
    # The rank of M_{4,4,5} is at most 65 by this construction.
    # Is there a known lower rank? I don't think so for these small dimensions.
    
    # Let me construct the full decomposition.
    
    # First, construct the 4x4*4x4 part using Strassen.
    # We use 2x2 blocks. A(4x4) = [[A11, A12], [A21, A22]] where each is 2x2.
    # B(4x4) = [[B11, B12], [B21, B22]] where each is 2x2.
    # C(4x4) = A*B
    
    # The 4x4*4x4 tensor can be decomposed using 7 "Strassen products" at the
    # 2x2 block level, and each such product is a 2x2*2x2 multiplication with
    # rank 7. So total rank = 7*7 = 49.
    
    # Let me construct this properly.
    
    # The base Strassen decomposition for 2x2*2x2:
    sU, sV, sW = strassen_2x2_decomp()
    
    # Now, for 4x4*4x4, we have 4x4 = 2x(2x2) blocks.
    # A has indices (i, j) where i, j in {0,1,2,3}.
    # Block indices: (bi, bj) where bi, bj in {0,1}, and within-block indices (si, sj) in {0,1}.
    # So A[i,j] = A[bi][bj][si][sj] where i = bi*2+si, j = bj*2+sj.
    
    # The 4x4*4x4 multiplication tensor T[i,j,k] (where i,j for A, k for B's columns)
    # can be written as sum over block-level Strassen products.
    
    # At the block level, we have 7 products M1..M7, each of which is a product
    # of linear combinations of 2x2 blocks. Each M_m is itself a 2x2*2x2 
    # multiplication (of the block entries).
    
    # Let me define the block-level Strassen coefficients.
    # For the block-level 2x2*2x2:
    # A_blocks = [[A00, A01], [A10, A11]] (each 2x2)
    # B_blocks = [[B00, B01], [B10, B11]] (each 2x2)
    # C_blocks = A_blocks * B_blocks
    
    # The 7 block-level Strassen products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # Each M_m is a 2x2*2x2 multiplication. So M_m = sum_{c=0}^{6} U_m[c] * V_m[c] * W_m[c]
    # where U_m, V_m, W_m are from the base Strassen decomposition.
    
    # The full 4x4*4x4 decomposition:
    # T[i,j,k] = sum_{m=0}^{6} sum_{c=0}^{6} coeff_A[m][i_block, i_sub, j_block, j_sub] * 
    #              coeff_B[m][j_block, j_sub, k_block, k_sub] * 
    #              coeff_C[m][k_block, k_sub, i_block, i_sub]
    
    # This is getting complex. Let me just build the full 4x4*4x4 decomposition
    # by combining the block-level and within-block Strassen decompositions.
    
    # Actually, let me take a different approach. I'll directly construct the
    # rank-49 decomposition for 4x4*4x4 by using the tensor product structure.
    
    # The 4x4*4x4 tensor is the tensor product of two 2x2*2x2 tensors in a certain way.
    # Specifically, if we view 4x4 matrices as 2x2 blocks of 2x2 matrices, then
    # the multiplication is a bilinear map from (2x2 blocks) x (2x2 blocks) -> (2x2 blocks).
    
    # The rank-49 decomposition can be written as:
    # For each block-level Strassen product m (7 of them) and each within-block
    # Strassen product c (7 of them), we have a rank-1 term.
    
    # Let me define the coefficients more carefully.
    
    # Block-level Strassen for A (4x4 as 2x2 blocks of 2x2):
    # A is indexed by (bi, bj, si, sj) where bi, bj in {0,1} (block), si, sj in {0,1} (sub).
    # A[i, j] where i = bi*2+si, j = bj*2+sj.
    
    # The block-level Strassen products involve linear combinations of blocks:
    # For product m, the A-coefficient for block (bi, bj) is a_m[bi][bj].
    # Then the within-block part is a 2x2*2x2 multiplication.
    
    # Let me just compute this explicitly.
    
    # Define block-level Strassen coefficients for A:
    # a_coeffs[m][bi][bj] for m in 0..6, bi in 0..1, bj in 0..1
    a_block_coeffs = [
        # M1: (A00+A11)
        [[1, 0], [0, 1]],
        # M2: (A10+A11)
        [[0, 0], [1, 1]],
        # M3: A00
        [[1, 0], [0, 0]],
        # M4: A11
        [[0, 0], [0, 1]],
        # M5: (A00+A01)
        [[1, 1], [0, 0]],
        # M6: (A10-A00)
        [[-1, 0], [1, 0]],
        # M7: (A01-A11)
        [[0, 1], [0, -1]],
    ]
    
    # Block-level Strassen coefficients for B:
    # b_coeffs[m][bj][bk] for m in 0..6, bj in 0..1, bk in 0..1
    b_block_coeffs = [
        # M1: (B00+B11)
        [[1, 0], [0, 1]],
        # M2: B00
        [[1, 0], [0, 0]],
        # M3: (B01-B11)
        [[0, 1], [0, -1]],
        # M4: (B10-B00)
        [[-1, 0], [1, 0]],
        # M5: B11
        [[0, 0], [0, 1]],
        # M6: (B00+B01)
        [[1, 1], [0, 0]],
        # M7: (B10+B11)
        [[0, 0], [1, 1]],
    ]
    
    # Block-level Strassen coefficients for C:
    # c_coeffs[m][ck][ci] for m in 0..6, ck in 0..1, ci in 0..1
    # C = A*B, so C_blocks = A_blocks * B_blocks
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    c_block_coeffs = [
        # M1: C11 += 1, C22 += 1
        [[1, 0], [0, 1]],
        # M2: C21 += 1, C22 -= 1
        [[0, 1], [0, -1]],
        # M3: C12 += 1, C22 += 1
        [[0, 0], [1, 1]],
        # M4: C11 += 1, C21 += 1
        [[1, 0], [1, 0]],
        # M5: C11 -= 1, C12 += 1
        [[-1, 0], [1, 0]],
        # M6: C22 += 1
        [[0, 0], [0, 1]],
        # M7: C11 += 1
        [[1, 0], [0, 0]],
    ]
    
    # Now, the within-block part: each block-level product m gives a 2x2*2x2
    # multiplication, which has the base Strassen decomposition sU, sV, sW.
    
    # The full 4x4*4x4 decomposition has rank 7*7 = 49.
    # U matrix: 16 rows (4x4), 49 columns
    # V matrix: 16 rows (4x4), 49 columns  
    # W matrix: 16 rows (4x4), 49 columns
    
    # For column index col = m*7 + c (m in 0..6, c in 0..6):
    # U[i, col] where i = bi*2+si:
    #   U[i, col] = a_block_coeffs[m][bi][bj] * sU[si*2+sj, c]
    #   Wait, but this sums over bj and sj. Let me reconsider.
    
    # Actually, the A block (bi, bj) is a 2x2 matrix with entries A[bi*2+si, bj*2+sj].
    # The block-level coefficient a_block_coeffs[m][bi][bj] multiplies the entire block.
    # Then the within-block Strassen decomposition applies to the 2x2 block.
    
    # So for a given block-level product m and within-block product c:
    # The contribution to A is: sum_{bj, sj} a_block_coeffs[m][bi][bj] * sU[sj*2+sj, c] * A[bi*2+si, bj*2+sj]
    # Wait, I'm confusing myself. Let me be more careful.
    
    # The A matrix has entries A[i, j] where i, j in {0,1,2,3}.
    # Write i = bi*2 + si, j = bj*2 + sj.
    # The block A[bi][bj] is the 2x2 matrix with entries A[bi*2+si, bj*2+sj] for si, sj in {0,1}.
    
    # The block-level Strassen product m uses linear combinations of blocks:
    # A_m = sum_{bj} a_block_coeffs[m][bi][bj] * A[bi][bj]
    # This is a 2x2 matrix (for fixed bi) with entries:
    # A_m[si, sj] = sum_{bj} a_block_coeffs[m][bi][bj] * A[bi*2+si, bj*2+sj]
    
    # Similarly for B:
    # B_m = sum_{bk} b_block_coeffs[m][bj][bk] * B[bj][bk]
    # B_m[sj, sk] = sum_{bk} b_block_coeffs[m][bj][bk] * B[bj*2+sj, bk*2+sk]
    
    # Then the within-block Strassen product c computes:
    # C_m[ci, ck] = sum_{si, sj} A_m[si, sj] * B_m[sj, sk] ... 
    # Actually, the within-block multiplication is:
    # C_block = A_m * B_m (2x2 * 2x2)
    # Using the base Strassen decomposition:
    # C_block[ck, ci] = sum_c sU[si*2+sj, c] * sV[sj*2+sk, c] * sW[sk*2+si, c] * A_m[si, sj] * B_m[sj, sk]
    
    # This is getting very complex. Let me just build the full decomposition
    # by iterating over all combinations.
    
    # Actually, let me step back and use a simpler approach.
    # I'll just use the schoolbook decomposition for now, which has rank 80.
    # The Strassen-based approach would give rank 65, but it's complex to implement.
    #
    # Given the time constraints, let me implement the Strassen-based approach
    # for the 4x4*4x4 part (rank 49) and schoolbook for the 4x4*4x1 part (rank 16).
    
    # Let me construct the 4x4*4x4 Strassen decomposition properly.
    
    # For the 4x4*4x4 part:
    # U44 shape: (16, 49)
    # V44 shape: (16, 49)
    # W44 shape: (16, 49)
    
    U44 = [[0]*49 for _ in range(16)]
    V44 = [[0]*49 for _ in range(16)]
    W44 = [[0]*49 for _ in range(16)]
    
    for m in range(7):
        for c in range(7):
            col = m * 7 + c
            # For each bi, si (row of U44):
            for bi in range(2):
                for si in range(2):
                    i = bi * 2 + si
                    for bj in range(2):
                        for sj in range(2):
                            j = bj * 2 + sj
                            # U44[i, col] += a_block_coeffs[m][bi][bj] * sU[si*2+sj, c]
                            U44[i][col] += a_block_coeffs[m][bi][bj] * sU[si * 2 + sj][c]
                    
            # For each bj, sj (row of V44):
            for bj in range(2):
                for sj in range(2):
                    j = bj * 2 + sj
                    for bk in range(2):
                        for sk in range(2):
                            k = bk * 2 + sk
                            # V44[j, col] += b_block_coeffs[m][bj][bk] * sV[sj*2+sk, c]
                            V44[j][col] += b_block_coeffs[m][bj][bk] * sV[sj * 2 + sk][c]
                    
            # For each ck, sk (row of W44):
            for ck in range(2):
                for sk in range(2):
                    k = ck * 2 + sk
                    for ci in range(2):
                        for si in range(2):
                            i = ci * 2 + si
                            # W44[k, col] += c_block_coeffs[m][ck][ci] * sW[sk*2+si, c]
                            W44[k][col] += c_block_coeffs[m][ck][ci] * sW[sk * 2 + si][c]
    
    # Now for the 4x4*4x1 part (the 5th column of B):
    # This is just A(4x4) * b(4x1) = c(4x1)
    # Schoolbook: rank 16
    # U shape: (16, 16), V shape: (4, 16), W shape: (4, 16)
    # Wait, the dimensions are:
    # A is 4x4, so U has 4*4=16 rows
    # b is 4x1, so V has 4*1=4 rows
    # c is 1x4, so W has 1*4=4 rows
    # But the interface says U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For the full problem: n=4, m=4, p=5
    # U: (4*4, rank) = (16, rank)
    # V: (4*5, rank) = (20, rank)
    # W: (5*4, rank) = (20, rank)
    
    # So for the 4x4*4x1 part, it's embedded in the full structure.
    # The 5th column of B is B[:, 4], which is a 4x1 vector.
    # The corresponding part of V is V[j*5+4, col] for j in 0..3.
    # The corresponding part of W is W[4*4+i, col] for i in 0..3 (since k=4).
    
    # Let me construct the full decomposition.
    # Total rank = 49 + 16 = 65
    
    total_rank = 49 + 16
    n, m, p = DIMENSIONS
    U = [[0]*total_rank for _ in range(n * m)]
    V = [[0]*total_rank for _ in range(m * p)]
    W = [[0]*total_rank for _ in range(p * n)]
    
    # Copy the 4x4*4x4 part (columns 0..48)
    for i in range(16):
        for col in range(49):
            U[i][col] = U44[i][col]
            V[i][col] = V44[i][col]
            W[i][col] = W44[i][col]
    
    # Add the 4x4*4x1 part (columns 49..64)
    # This is A(4x4) * b(4x1) where b = B[:, 4]
    # Schoolbook: for each i, j: C[i] += A[i,j] * B[j, 4]
    # In the tensor: U[i*4+j, col] = A[i,j], V[j*5+4, col] = B[j,4], W[4*4+i, col] = C[i]
    col_offset = 49
    for i in range(4):
        for j in range(4):
            col = col_offset + i * 4 + j
            U[i * 4 + j][col] = 1
            V[j * 5 + 4][col] = 1
            W[4 * 4 + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
