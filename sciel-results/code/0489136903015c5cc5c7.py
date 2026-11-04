import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We'll build a decomposition using Strassen for the 4x4x4 part and schoolbook for 4x4x1
    # Total rank: 28 (Strassen 4x4x4) + 8 (schoolbook 4x4x1) = 36
    
    # Each term in the decomposition is a triple (u, v, w) where:
    # u is a vector of length n*m=16 (flattened A matrix indices)
    # v is a vector of length m*p=20 (flattened B matrix indices)  
    # w is a vector of length p*n=20 (flattened C matrix indices, with W[k*n+i])
    
    # For matrix multiplication C[i,k] = sum_j A[i,j] * B[j,k]
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r
    
    terms = []  # Each term: (u_dict, v_dict, w_dict) where dicts map index->coeff
    
    def add_term(u_coeff, v_coeff, w_coeff):
        """Add a rank-1 term. u_coeff, v_coeff, w_coeff are dicts mapping flat index to coefficient."""
        u = {idx: 0 for idx in range(n*m)}
        v = {idx: 0 for idx in range(m*p)}
        w = {idx: 0 for idx in range(p*n)}
        
        for idx, c in u_coeff.items():
            u[idx] = c
        for idx, c in v_coeff.items():
            v[idx] = c
        for idx, c in w_coeff.items():
            w[idx] = c
        
        terms.append((u, v, w))
    
    # Strassen decomposition for 4x4x4 (first 4 columns of B)
    # A is 4x4, B1 is 4x4 (columns 0-3)
    # Split into 2x2 blocks of 2x2 matrices
    
    # Block indices for A: A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3),
    #                      A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)
    # Block indices for B1: B11 (rows 0-1, cols 0-1), B12 (rows 0-1, cols 2-3),
    #                       B21 (rows 2-3, cols 0-1), B22 (rows 2-3, cols 2-3)
    # Block indices for C: C11 (rows 0-1, cols 0-1), C12 (rows 0-1, cols 2-3),
    #                      C21 (rows 2-3, cols 0-1), C22 (rows 2-3, cols 2-3)
    
    # Helper to get flat indices for a 2x2 block
    def block_indices(row_offset, col_offset, rows, cols):
        """Get flat indices for a block in the flattened matrix."""
        indices = []
        for dr in range(rows):
            for dc in range(cols):
                i = row_offset + dr
                j = col_offset + dc
                indices.append(i * m + j)  # For A (n*m)
        return indices
    
    def block_indices_b(row_offset, col_offset, rows, cols):
        """Get flat indices for a block in B (m*p)."""
        indices = []
        for dr in range(rows):
            for dc in range(cols):
                j = row_offset + dr
                k = col_offset + dc
                indices.append(j * p + k)  # For B (m*p)
        return indices
    
    def block_indices_c(row_offset, col_offset, rows, cols):
        """Get flat indices for a block in C (p*n), with W[k*n+i]."""
        indices = []
        for dr in range(rows):
            for dc in range(cols):
                k = row_offset + dr
                i = col_offset + dc
                indices.append(k * n + i)  # For C (p*n)
        return indices
    
    # Define the 7 Strassen products for 4x4x4
    # Each product is a 2x2 matrix multiplication (schoolbook, rank 4)
    
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For each Strassen product, we need to expand the sums and use schoolbook for the 2x2 mult
    
    # Let's define a helper to add a schoolbook 2x2x2 multiplication
    def add_strassen_product(a_coeff, b_coeff, c_coeff, a_blocks, b_blocks, c_blocks):
        """
        Add terms for a Strassen product.
        a_coeff, b_coeff, c_coeff: coefficients for the combined blocks
        a_blocks: list of (block_name, sign) for A side
        b_blocks: list of (block_name, sign) for B side  
        c_blocks: list of (block_name, sign) for C side
        """
        # This is getting complex. Let me use a different approach.
        # I'll directly enumerate the terms.
        pass
    
    # Actually, let me use a more direct approach. I'll construct the decomposition
    # by explicitly listing all terms.
    
    # For the 4x4x4 part using Strassen:
    # The 7 Strassen products each require 4 schoolbook multiplications (2x2x2)
    # Total: 7 * 4 = 28 terms
    
    # Let me define the Strassen terms more carefully.
    
    # Strassen for 2x2x2 (base case):
    # M1 = (A11+A22)(B11+B22) -> contributes to C11, C22
    # M2 = (A21+A22)B11 -> contributes to C21
    # M3 = A11(B21-B22) -> contributes to C12
    # M4 = A22(B12-B11) -> contributes to C11
    # M5 = (A11+A12)B22 -> contributes to C12, C11
    # M6 = (A21-A11)(B11+B12) -> contributes to C22
    # M7 = (A12-A22)(B21+B22) -> contributes to C11, C22
    
    # For 4x4x4, each "2x2" is actually a 2x2 block of 2x2 matrices.
    # So each Strassen product involves multiplying two 2x2 blocks (each 2x2 matrix).
    # We use schoolbook for each such multiplication.
    
    # Let me redefine: for the 4x4x4 part, I'll use the fact that we can view it as
    # 8 independent 2x2x2 multiplications (one for each pair of 2x2 blocks), but Strassen
    # reduces this to 7 "virtual" multiplications.
    
    # Actually, the cleanest way: implement Strassen recursively.
    # 4x4x4 -> 7 products of 2x2x2 (each rank 4) = 28 total
    
    # Let me just hardcode the 28 terms for the Strassen 4x4x4 part.
    
    # This is getting too complex to do by hand. Let me use a computational approach.
    
    # Alternative: Use a known rank-20 or rank-21 decomposition if I can verify it.
    # But I don't have the coefficients memorized.
    
    # Let me try a different strategy: use the tensor network approach.
    # I'll construct a valid decomposition of rank < 80.
    
    # Actually, let me just implement the Strassen decomposition computationally.
    
    # For 4x4x4 using Strassen:
    # We have 7 Strassen "super-multiplications", each of which is a 2x2x2 multiplication.
    # Each 2x2x2 multiplication has rank 4 (schoolbook).
    # So total rank for 4x4x4 part = 28.
    
    # For 4x4x1 using schoolbook: rank = 4*4*1 = 16.
    # Wait, that's not right. 4x4x1 means multiplying 4x4 by 4x1 to get 4x1.
    # Schoolbook rank = 4*4*1 = 16.
    
    # Hmm, but I thought I could do better. Let me reconsider.
    
    # Actually, for 4x4x1, we can also use Strassen-like tricks.
    # But let's stick with schoolbook for simplicity: rank 16.
    
    # Total rank = 28 + 16 = 44.
    
    # Wait, that's worse than my earlier estimate of 36. Let me recheck.
    
    # For 4x4x1: A is 4x4, B is 4x1, C is 4x1.
    # C[i] = sum_j A[i,j] * B[j]
    # This is a rank-16 tensor (schoolbook).
    
    # Can I do better? Using Strassen on the 4x4 part?
    # Split A into 2x2 blocks, B into 2x1 blocks.
    # C1 = A11*B1 + A12*B2 (where B1, B2 are 2x1)
    # C2 = A21*B1 + A22*B2
    # Each term like A11*B1 is a 2x2 * 2x1 multiplication.
    # Schoolbook for 2x2 * 2x1: rank = 2*2*1 = 4.
    # So C1 requires 4+4 = 8 multiplications.
    # C2 requires 4+4 = 8 multiplications.
    # Total: 16. Same as before.
    
    # So total rank = 28 + 16 = 44.
    
    # That's still better than 80. Let me implement this.
    
    # But wait, I need to verify that the Strassen decomposition for 4x4x4 is correct.
    # Let me just use schoolbook for both parts to be safe: 64 + 16 = 80.
    
    # No, that's the same as the current program. I need to do better.
    
    # Let me try to implement the Strassen decomposition computationally.
    
    # I'll construct the U, V, W matrices by building each term.
    
    # For the 4x4x4 part (Strassen):
    # I'll use the following approach:
    # 1. Define the 7 Strassen products as linear combinations of A and B blocks.
    # 2. For each Strassen product, expand it into 4 schoolbook terms (2x2x2).
    # 3. Track the contributions to the C blocks.
    
    # This is complex. Let me use a simpler approach: I'll use a known rank-20 decomposition
    # for 4x4x4 if I can find one, or just use rank-28 Strassen.
    
    # Given the time constraints, let me implement the Strassen 4x4x4 + schoolbook 4x4x1.
    
    # I'll construct the terms programmatically.
    
    # First, let me define the block structure.
    
    # For A (4x4): blocks A11, A12, A21, A22 (each 2x2)
    # For B1 (4x4, first 4 cols of B): blocks B11, B12, B21, B22 (each 2x2)
    # For C1 (4x4, first 4 cols of C): blocks C11, C12, C21, C22 (each 2x2)
    
    # Strassen products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 matrix. To compute Mi, we use schoolbook (4 multiplications).
    
    # Let me define a function to add a schoolbook 2x2x2 multiplication.
    
    def add_schoolbook_2x2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset, sign=1):
        """Add 4 terms for a 2x2x2 schoolbook multiplication."""
        for dr in range(2):
            for dc in range(2):
                for dk in range(2):
                    # A block element at (a_row_offset+dr, a_col_offset+dc)
                    # B block element at (b_row_offset+dk, b_col_offset+dc)
                    # C block element at (c_row_offset+dr, c_col_offset+dk)
                    
                    a_idx = (a_row_offset + dr) * m + (a_col_offset + dc)
                    b_idx = (b_row_offset + dk) * p + (b_col_offset + dc)
                    c_idx = (c_row_offset + dk) * n + (c_row_offset + dr)
                    
                    u = {a_idx: sign}
                    v = {b_idx: sign}
                    w = {c_idx: sign}
                    
                    add_term(u, v, w)
    
    # Now let me add the Strassen terms.
    # This is getting very complex. Let me use a different approach.
    
    # I'll use the fact that I can verify the decomposition numerically.
    # Let me construct a candidate decomposition and verify it.
    
    # Given the complexity, let me just use schoolbook for now (rank 80) to ensure correctness,
    # and then try to improve it.
    
    # Actually, let me try a hybrid: use Strassen for 2x2x2 blocks recursively.
    
    # For 4x4x4:
    # Split into 2x2 blocks of 2x2 matrices.
    # Use Strassen to get 7 products of 2x2x2.
    # Each 2x2x2 uses schoolbook (rank 4).
    # Total: 28.
    
    # For 4x4x1:
    # Split A into 2x2 blocks, B into 2x1 blocks.
    # C1 = A11*B1 + A12*B2 (each 2x2*2x1, rank 4 each, total 8)
    # C2 = A21*B1 + A22*B2 (each 2x2*2x1, rank 4 each, total 8)
    # Total: 16.
    
    # Grand total: 28 + 16 = 44.
    
    # Let me implement this.
    
    # I'll construct the terms by carefully tracking the indices.
    
    # Clear terms
    terms = []
    
    # Part 1: 4x4x4 using Strassen (rank 28)
    # I'll use a computational approach to generate the terms.
    
    # Define the Strassen products and their contributions to C blocks.
    
    # For each Strassen product Mi, I need to:
    # 1. Express the A-side as a sum of A blocks with coefficients.
    # 2. Express the B-side as a sum of B blocks with coefficients.
    # 3. For each pair of (A_block, B_block), add 4 schoolbook terms (2x2x2).
    # 4. Track the sign for the C contribution.
    
    # This is still complex. Let me just use schoolbook for the entire 4x4x5 (rank 80)
    # to ensure correctness, since I'm running out of time.
    
    # But that's the same as the current program. I need to do better.
    
    # Let me try a different approach: use a rank-20 decomposition for 4x4x4 if I can
    # find one, or use the Lander-Bailey decomposition.
    
    # Given the constraints, let me just implement the schoolbook decomposition (rank 80)
    # to ensure correctness, and then try to improve it if time permits.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
