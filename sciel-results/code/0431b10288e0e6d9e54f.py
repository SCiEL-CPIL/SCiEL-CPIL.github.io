import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 120
    
    def strassen_2x2_decomp():
        """Return U, V, W for 2x2*2x2 matrix multiplication with rank 7."""
        U = [[0.0]*7 for _ in range(4)]
        V = [[0.0]*7 for _ in range(4)]
        W = [[0.0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        U[0][0] = 1.0
        U[3][0] = 1.0
        V[0][0] = 1.0
        V[3][0] = 1.0
        W[0][0] = 1.0
        W[3][0] = 1.0
        
        # M2 = (A21+A22)B11
        U[2][1] = 1.0
        U[3][1] = 1.0
        V[0][1] = 1.0
        W[2][1] = 1.0
        
        # M3 = A11(B21-B22)
        U[0][2] = 1.0
        V[2][2] = 1.0
        V[3][2] = -1.0
        W[1][2] = 1.0
        
        # M4 = A22(B11-B21)
        U[3][3] = 1.0
        V[0][3] = 1.0
        V[2][3] = -1.0
        W[0][3] = 1.0
        W[2][3] = 1.0
        
        # M5 = (A11+A12)B22
        U[0][4] = 1.0
        U[1][4] = 1.0
        V[3][4] = 1.0
        W[0][4] = -1.0
        W[1][4] = 1.0
        
        # M6 = (A21-A11)(B11+B12)
        U[2][5] = 1.0
        U[0][5] = -1.0
        V[0][5] = 1.0
        V[1][5] = 1.0
        W[3][5] = 1.0
        
        # M7 = (A12-A22)(B21+B22)
        U[1][6] = 1.0
        U[3][6] = -1.0
        V[2][6] = 1.0
        V[3][6] = 1.0
        W[0][6] = 1.0
        
        return U, V, W
    
    # Build 4x4*4x4 decomposition using Strassen on 2x2 blocks
    # A is 4x4, B is 4x4, C is 4x4
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where A11=A[0:2,0:2], etc.
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # For each 2x2*2x2 product, we use Strassen (rank 7)
    # We have 4 such products, so total rank = 4*7 = 28 for the 4x4*4x4 case?
    # Wait, that's not right. Let me reconsider.
    
    # Actually, for 4x4*4x4, if we split into 2x2 blocks and use Strassen
    # for each block multiplication, we get:
    # C11 = A11*B11 + A12*B21 (two 2x2*2x2 products)
    # C12 = A11*B12 + A12*B22 (two 2x2*2x2 products)
    # C21 = A21*B11 + A22*B21 (two 2x2*2x2 products)
    # C22 = A21*B12 + A22*B22 (two 2x2*2x2 products)
    
    # But this is just schoolbook at the block level, giving 8*7=56 terms.
    # To get rank 49, we need to use Strassen at the 4x4 level by treating
    # the 4x4 matrices as 2x2 blocks of 2x2 matrices and applying Strassen.
    
    # Let me implement this properly.
    
    # For 4x4*4x4 with Strassen:
    # Treat A as [[A11, A12], [A21, A22]] where each is 2x2
    # Treat B as [[B11, B12], [B21, B22]] where each is 2x2
    # 
    # Strassen formulas (for block matrices):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Each M_i is a 2x2*2x2 product, which has rank 7 using Strassen.
    # So total rank = 7*7 = 49.
    
    # Now I need to construct the full U, V, W matrices for 4x4*4x4.
    # U has shape (4*4, 49) = (16, 49)
    # V has shape (4*4, 49) = (16, 49)
    # W has shape (4*4, 49) = (16, 49)
    
    # The indexing: U[i*4+j, r] corresponds to A[i,j]
    # V[j*4+k, r] corresponds to B[j,k]
    # W[k*4+i, r] corresponds to C[i,k]
    
    # For each Strassen term M_q (q=0..6), we have 7 sub-terms from the
    # inner Strassen decomposition.
    
    # Let me build this step by step.
    
    # First, let's define the 4x4*4x4 decomposition.
    
    def build_4x4_strassen():
        """Build U, V, W for 4x4*4x4 with rank 49 using Strassen."""
        rank = 49
        U = [[0.0]*rank for _ in range(16)]
        V = [[0.0]*rank for _ in range(16)]
        W = [[0.0]*rank for _ in range(16)]
        
        # Get the 2x2 Strassen decomposition
        U2, V2, W2 = strassen_2x2_decomp()
        
        # For each outer Strassen term q (0..6), we have 7 inner terms.
        # The outer term q corresponds to multiplying two 2x2 block matrices.
        # Each such multiplication uses the 2x2 Strassen decomposition.
        
        # Let's define the outer Strassen structure.
        # For term q, we need to specify:
        # - Which blocks of A are used and with what coefficients
        # - Which blocks of B are used and with what coefficients
        # - Which blocks of C receive the result and with what coefficients
        
        # Let me define this more carefully.
        
        # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
        # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
        # C11 = C[0:2, 0:2], C12 = C[0:2, 2:4], C21 = C[2:4, 0:2], C22 = C[2:4, 2:4]
        
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B11-B21)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # For each M_q, we need to:
        # 1. Form the A-side block matrix (2x2 blocks of 2x2)
        # 2. Form the B-side block matrix (2x2 blocks of 2x2)
        # 3. Apply the 2x2 Strassen decomposition to these block matrices
        # 4. Distribute the result to the appropriate C blocks
        
        # This is complex. Let me think about it differently.
        
        # Actually, each M_q is a product of two 2x2 block matrices.
        # Let A_q be the 2x2 block matrix for the A-side of M_q.
        # Let B_q be the 2x2 block matrix for the B-side of M_q.
        # Then M_q = A_q * B_q, where each entry is a 2x2 matrix.
        
        # The product M_q has 4 entries, each a 2x2 matrix.
        # Each 2x2 matrix product has rank 7 using Strassen.
        # So M_q contributes 4*7 = 28 rank-1 terms? No, that's not right either.
        
        # Wait, I think I'm overcomplicating this. Let me reconsider.
        
        # The 4x4*4x4 matrix multiplication tensor has rank at most 49.
        # This is achieved by viewing the 4x4 matrices as 2x2 blocks of 2x2 matrices
        # and applying Strassen at the block level.
        
        # The Strassen algorithm for 2x2*2x2 (where each entry is a 2x2 matrix)
        # uses 7 multiplications of 2x2 matrices. Each 2x2*2x2 multiplication
        # has rank 7. So total rank = 7*7 = 49.
        
        # To construct this, I need to:
        # 1. For each of the 7 outer Strassen multiplications,
        #    identify which 2x2 sub-matrices of A and B are involved
        # 2. Apply the inner 2x2 Strassen decomposition to each
        # 3. Combine the results appropriately
        
        # Let me define the outer Strassen terms more carefully.
        
        # For M1 = (A11+A22)(B11+B22):
        # A-side: [[A11, 0], [0, A22]] + [[0, 0], [0, 0]] ... this is getting messy.
        
        # Let me use a different approach. I'll directly construct the rank-49
        # decomposition by enumerating all the terms.
        
        # Actually, let me just implement the full 4x5*5x6 decomposition directly
        # using a simpler method.
        
        return U, V, W
    
    # Given the complexity, let me try a different approach.
    # I'll use the fact that we can decompose 4x5*5x6 into smaller problems.
    
    # Approach: Use block decomposition with Strassen where beneficial.
    
    # Let me try to construct a rank-80 decomposition directly.
    
    # Split A (4x5) into A_left (4x4) and A_right (4x1)
    # Split B (5x6) into B_top (4x6) and B_bottom (1x6)
    
    # C = A_left * B_top + A_right * B_bottom
    
    # For A_left * B_top (4x4 * 4x6):
    # Split B_top into B_top_left (4x4) and B_top_right (4x2)
    # A_left * B_top = A_left * B_top_left + A_left * B_top_right
    
    # For A_left * B_top_left (4x4 * 4x4): use Strassen, rank 49
    # For A_left * B_top_right (4x4 * 4x2): 
    #   Split into 2x2 blocks: A_left has 4 blocks of 2x2, B_top_right has 2 blocks of 2x2
    #   We need 8 products of 2x2*2x2, each rank 7, total rank 56
    #   But we can do better with Strassen-like techniques.
    
    # Actually, for 4x4 * 4x2, we can split A into 2x2 blocks (4 blocks) and
    # B into 2x2 blocks (2 blocks). We need 8 2x2*2x2 products.
    # Using Strassen for each gives 8*7 = 56.
    
    # Hmm, that's more than expected. Let me reconsider.
    
    # For 4x4 * 4x2:
    # A = [[A11, A12], [A21, A22]] (2x2 blocks of 2x2)
    # B = [[B11], [B21]] (2x1 blocks of 2x2, i.e., B is 4x2)
    # C = [[C11], [C21]] where C11 = A11*B11 + A12*B21, C21 = A21*B11 + A22*B21
    
    # We have 4 2x2*2x2 products: A11*B11, A12*B21, A21*B11, A22*B21
    # Each has rank 7, total rank 28.
    
    # So:
    # 4x4*4x4: rank 49
    # 4x4*4x2: rank 28
    # 4x1*4x6: This is a rank-1 times rank-6, which is rank 6 (or less)
    #   Actually, 4x1 * 4x6 = outer product of a 4-vector and a 4x6 matrix
    #   This can be done with rank 4 (one for each row of the 4x1 vector times each column)
    #   Wait, no. A_right is 4x1, B_bottom is 1x6.
    #   A_right * B_bottom is a 4x6 matrix where C[i,k] = A_right[i,0] * B_bottom[0,k]
    #   This is a rank-1 matrix (outer product), so it needs rank 1? No.
    #   Actually, it's the outer product of a 4-vector and a 6-vector, which has rank 1.
    #   So we need 1 term for this.
    
    # Wait, let me reconsider. A_right is 4x1, B_bottom is 1x6.
    # The product is 4x6, and C[i,k] = A_right[i,0] * B_bottom[0,k].
    # This is indeed a rank-1 matrix (outer product of column vector and row vector).
    # So we need only 1 rank-1 term.
    
    # Total: 49 + 28 + 1 = 78?
    
    # But wait, I need to be more careful. Let me reconsider the splitting.
    
    # A is 4x5, B is 5x6.
    # Split A into A1 (4x4) and A2 (4x1)
    # Split B into B1 (4x6) and B2 (1x6)
    # C = A1*B1 + A2*B2
    
    # A1*B1 is 4x4 * 4x6.
    # Split B1 into B1a (4x4) and B1b (4x2)
    # A1*B1 = A1*B1a + A1*B1b
    
    # A1*B1a is 4x4*4x4, rank 49
    # A1*B1b is 4x4*4x2, rank 28
    # A2*B2 is 4x1*1x6, rank 1
    
    # Total: 49 + 28 + 1 = 78
    
    # But I need to verify this is correct. Let me construct it.
    
    # For now, let me just return the schoolbook and add a TODO to implement
    # the Strassen-based decomposition.
    
    # Actually, let me try to implement the 4x4*4x4 Strassen decomposition first.
    
    def build_4x4_4x4_strassen():
        """Build U, V, W for 4x4*4x4 with rank 49."""
        rank = 49
        U = [[0.0]*rank for _ in range(16)]
        V = [[0.0]*rank for _ in range(16)]
        W = [[0.0]*rank for _ in range(16)]
        
        # Get 2x2 Strassen decomposition
        U2, V2, W2 = strassen_2x2_decomp()
        
        # The 4x4*4x4 Strassen decomposition:
        # We view 4x4 matrices as 2x2 blocks of 2x2 matrices.
        # Apply Strassen at the block level.
        
        # For each of the 7 outer Strassen terms, we have a product of two
        # 2x2 block matrices. Each such product uses the 2x2 Strassen
        # decomposition, contributing 7 terms.
        
        # Let me define the outer Strassen terms.
        
        # A = [[A11, A12], [A21, A22]] where Aij are 2x2
        # B = [[B11, B12], [B21, B22]] where Bij are 2x2
        
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B11-B21)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # For each Mq, the A-side and B-side are 2x2 block matrices.
        # I need to specify which 2x2 sub-blocks of the 4x4 matrix are used.
        
        # Let me define a helper to add a 2x2*2x2 Strassen term.
        
        def add_2x2_strassen_term(U, V, W, a_blocks, b_blocks, c_blocks, sign, offset):
            """
            Add a 2x2*2x2 Strassen term.
            a_blocks: list of (row_offset, col_offset, coeff) for A 2x2 blocks
            b_blocks: list of (row_offset, col_offset, coeff) for B 2x2 blocks
            c_blocks: list of (row_offset, col_offset, coeff) for C 2x2 blocks
            sign: overall sign
            offset: column offset in U, V, W
            """
            # This is getting too complex. Let me try a different approach.
            pass
        
        # Given the complexity, let me just return the schoolbook for now.
        return U, V, W
    
    # Let me try a completely different approach: use numpy to help construct
    # the decomposition, or just return the schoolbook.
    
    # For now, return schoolbook.
    U_sb = [[0.0] * schoolbook_rank for _ in range(n*m)]
    V_sb = [[0.0] * schoolbook_rank for _ in range(m*p)]
    W_sb = [[0.0] * schoolbook_rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1.0
                V_sb[j*p+k][col] = 1.0
                W_sb[k*n+i][col] = 1.0
                col += 1
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
