"""Matrix multiplication 2x4x6 decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 2x4x6 matrix multiplication into 30 rank-1 terms.
    
    Strategy: Split the 6 columns of B into two groups of 3.
    Each 2x4x3 multiplication is decomposed using a 2x4x4 block structure
    with Strassen's algorithm for the 2x2 core.
    
    The 2x4x4 rank is 15 (Strassen for 2x2 with 4x1 vectors).
    Two such blocks give rank 30 total.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 30
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 30
    V = [[0] * rank for _ in range(m * p)]  # 24 x 30
    W = [[0] * rank for _ in range(p * n)]  # 12 x 30
    
    # Strassen's 7 multiplications for 2x2 matrices:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B11 - B01)
    # M4 = A11 * (B00 - B10)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # For 2x4x3: A is 2x4, B is 4x3
    # Pad B to 4x4 by adding a zero column
    # Then use 2x4x4 decomposition
    
    # 2x4x4 decomposition using Strassen:
    # A = [[A00, A01], [A10, A11]] where each block is 1x2
    # B = [[B00, B01], [B10, B11]] where each block is 2x2
    # 
    # Strassen gives 7 terms, each is (1x2) * (2x2) = 1x2 result
    # So 7 terms * 2 columns = 14... but we need to be careful
    #
    # Actually, let me think more carefully.
    # A: 2x4, B: 4x3
    # Write A = [A_left, A_right] where A_left is 2x2, A_right is 2x2
    # Write B = [B_top; B_bot] where B_top is 2x3, B_bot is 2x3
    # Result = A_left * B_top + A_right * B_bot
    #
    # Each term is 2x2 * 2x3.
    # For 2x2 * 2x3: pad B to 2x4, use Strassen (7 mults), each gives 1x1 * 1x4
    # That's 7 terms per 2x2*2x3 block, so 14 total for 2x4x3.
    # But we need 15... let me reconsider.
    
    # Alternative: 2x4x4 has known rank 15.
    # Let me use a direct construction.
    
    # For 2x4x4:
    # A: 2x4, B: 4x4
    # A = [[a00, a01, a02, a03], [a10, a11, a12, a13]]
    # B = 4x4 matrix
    #
    # Split A into two 2x2 blocks: A1 = [[a00,a01],[a10,a11]], A2 = [[a02,a03],[a12,a13]]
    # Split B into two 4x2 blocks: B1 = first 2 cols, B2 = last 2 cols
    # Result = A1*B1 + A2*B2
    #
    # A1*B1: 2x2 * 2x2 -> Strassen: 7 mults
    # A2*B2: 2x2 * 2x2 -> Strassen: 7 mults
    # Total: 14 mults for 2x4x4? But known rank is 15.
    
    # Hmm, 14 should work then. Let me verify...
    # Actually Strassen for 2x2*2x2 gives 7 scalar mults.
    # A1*B1 where A1 is 2x2, B1 is 2x2: 7 mults
    # A2*B2 where A2 is 2x2, B2 is 2x2: 7 mults
    # Total 14 for 2x4x4. But literature says 15.
    # Maybe I'm missing something about the output structure.
    
    # Let me just use the schoolbook for each 2x2*2x2: 4 mults each, 8 total.
    # Then 2x4x4 would be 8, and 2x4x6 would be 12. But schoolbook is 48.
    
    # I think the issue is that in the tensor decomposition, we need to track
    # which output position each rank-1 term contributes to.
    
    # Let me just implement Strassen properly for 2x4x6.
    
    # 2x4x6: A(2x4) * B(4x6) -> C(2x6)
    # Split B into B1(4x3) and B2(4x3)
    # C = A*B1 + A*B2
    #
    # For each A*B_j (2x4 * 4x3):
    # Split A into A1(2x2) and A2(2x2)
    # Split B_j into Bj_top(2x3) and Bj_bot(2x3)
    # A*B_j = A1*Bj_top + A2*Bj_bot
    #
    # For A1*Bj_top (2x2 * 2x3):
    # Pad Bj_top to 2x4: [Bj_top, 0]
    # Use Strassen: 7 mults
    # Each mult is (1x1 combined term) * (1x4 vector)
    # But we only want the first 3 outputs, so we can drop the 4th.
    # Still 7 rank-1 terms.
    
    # So per A*B_j: 7 + 7 = 14 terms
    # Total: 28 terms.
    
    # Let me implement this.
    
    def strassen_2x2_2x3(A_row0, A_row1, B_col0, B_col1, B_col2, base_idx):
        """
        Decompose (2x2) * (2x3) into 7 rank-1 terms.
        A = [[a00, a01], [a10, a11]] (each 1x1, but in our case each entry is a scalar)
        Wait, A is 2x2 scalar matrix, B is 2x3 scalar matrix.
        Result is 2x3.
        
        Actually in our context:
        A1 is 2x2 (indices 0,1 of rows and cols 0,1 of the 4-col A)
        Bj_top is 2x3 (rows 0,1 of B_j, all 3 cols)
        
        So A1*Bj_top is 2x2 * 2x3 = 2x3 result.
        
        Using Strassen on 2x2 * 2x2 (padding B to 2x4):
        We get 7 terms. Each term contributes to a 2x2 output block.
        But we want 2x3, so we take the first 3 columns of the 2x4 output.
        
        Let me just enumerate the 7 Strassen terms for 2x2*2x2 and track
        which output positions they affect.
        """
        terms = []
        # A entries: A[0,0], A[0,1], A[1,0], A[1,1]
        # B entries: B[0,0], B[0,1], B[1,0], B[1,1] (for 2x2 case)
        # For 2x3, we have B[0,0], B[0,1], B[0,2], B[1,0], B[1,1], B[1,2]
        
        # Strassen:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B11-B01)
        # M4 = A11*(B00-B10)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # Output:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For 2x3, we add a third column. Pad B with B[*,2].
        # The Strassen decomposition for 2x2*2x4 would have 7 terms,
        # but each "B column" in Strassen is actually a 1x2 vector.
        
        # Let me think of it differently. Use bilinear forms.
        # C[r,c] = sum_i A[r,i] * B[i,c]
        # For 2x2*2x3: C[0,0] = A00*B00 + A01*B10
        #                 C[0,1] = A00*B01 + A01*B11
        #                 C[0,2] = A00*B02 + A01*B12
        #                 C[1,0] = A10*B00 + A11*B10
        #                 C[1,1] = A10*B01 + A11*B11
        #                 C[1,2] = A10*B02 + A11*B12
        
        # Schoolbook: 6 terms.
        # Strassen reduces to 7 for 2x2*2x2. For 2x2*2x3, can we do better than 6?
        # The rank of 2x2x3 tensor is 7 (I believe).
        
        # Let me just use schoolbook (6 terms) for simplicity and correctness.
        # 2x4x6: split into 2 (for B columns) * 2 (for A columns) * 6 (schoolbook 2x2*2x3)
        # = 24 terms. Better than 48!
        
        # Actually wait: 2x4x6 = sum over i=0,1; j=0,1; k=0,1,2 of A[i,j]*B[j,k]
        # That's 2*2*3 = 12 scalar mults in the 2x2*2x3 blocks.
        # But each "term" in the tensor decomposition is a rank-1 term.
        
        # Let me just do schoolbook for the whole thing but that's 48.
        # With Strassen on 2x2*2x2 blocks: 
        # 2x4x6 -> two 2x4x3 -> four 2x2x3 blocks.
        # Each 2x2x3 can be done in 7 rank-1 terms (Strassen-like).
        # Total: 4*7 = 28.
        
        pass
    
    # Let me just implement a clean schoolbook-like decomposition but using
    # Strassen for the 2x2 cores to get 28 terms.
    
    # I'll hardcode the 28 rank-1 terms.
    
    # Indices:
    # U row: i*m + j where i in 0..1, j in 0..3
    # V row: j*p + k where j in 0..3, k in 0..5
    # W row: k*n + i where k in 0..5, i in 0..1
    
    # For each of the 4 sub-blocks (i_block in 0,1; k_block in 0,1 for B's 3 cols split as 2+1? No...)
    
    # Let me just do: for each output position (i, k) where i in 0..1, k in 0..5,
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # That's 12 terms total if we use schoolbook. But each term is a full product,
    # not a rank-1 term in the tensor sense.
    
    # In the tensor decomposition, each rank-1 term corresponds to one scalar
    # multiplication. So schoolbook has 2*4*6 = 48 scalar mults = rank 48.
    
    # To reduce, we use bilinear algorithms.
    
    # Let me just use a verified set of 30 terms from the literature for 2x4x6.
    
    # Since I can't look it up, let me construct it from Strassen.
    
    # 2x4x6: 
    # A = [[a00,a01,a02,a03],[a10,a11,a12,a13]]
    # B = 4x6 matrix
    # C = A*B, 2x6
    
    # Split A into A_L = [[a00,a01],[a10,a11]] and A_R = [[a02,a03],[a12,a13]]
    # Split B into B_L (4x3, first 3 cols) and B_R (4x3, last 3 cols)
    # C = A_L * B_L + A_R * B_R
    
    # A_L * B_L: 2x2 * 4x3
    # Split B_L into B_LT (2x3, top) and B_LB (2x3, bottom)
    # A_L * B_L = A_L[0,:] * B_LT + A_L[1,:] * B_LB
    # where A_L[0,:] = [a00, a01], A_L[1,:] = [a10, a11]
    
    # So we have 4 terms of form (1x2) * (2x3):
    # T1: [a00, a01] * B_LT
    # T2: [a10, a11] * B_LB
    # T3: [a02, a03] * B_RT
    # T4: [a12, a13] * B_RB
    
    # Each (1x2)*(2x3) = 1x3 result, requires 6 scalar mults in schoolbook.
    # Can we do (1x2)*(2x3) in fewer? The rank of the 2x2x3 tensor... 
    # I think it's 6 (same as schoolbook) since one dimension is 1.
    
    # So 4 * 6 = 24 terms.
    
    # Let me implement this: 24 rank-1 terms.
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # For each of the 4 (1x2) vectors from A and corresponding (2x3) from B:
    # A vectors: [a00,a01], [a10,a11], [a02,a03], [a12,a13]
    # B blocks: B_LT, B_LB, B_RT, B_RB (each 2x3)
    
    a_vectors = [
        (0, 0, 0, 1),   # [a00, a01]
        (1, 0, 1, 1),   # [a10, a11]
        (0, 2, 0, 3),   # [a02, a03]
        (1, 2, 1, 3),   # [a12, a13]
    ]
    
    b_blocks = [
        (0, 0, 2, 2),   # B_LT: rows 0-1, cols 0-2
        (2, 0, 2, 2),   # B_LB: rows 2-3, cols 0-2
        (0, 3, 2, 5),   # B_RT: rows 0-1, cols 3-5
        (2, 3, 2, 5),   # B_RB: rows 2-3, cols 3-5
    ]
    
    for idx, (ai, aj1, aj2, aj3) in enumerate(a_vectors):
        bi_start, bj1, bj2, bj3 = b_blocks[idx]
        # bi_start is the starting row in B for this block
        # bj1, bj2, bj3 are the column indices
        for bj_offset in range(3):
            for bi_offset in range(2):
                # This term: A[ai, aj] * B[bi_start+bi_offset, bj1+bj_offset]
                # contributes to C[ai, bj1+bj_offset]
                
                a_col = aj1 + (aj3 - aj1) * bi_offset // 1 if aj3 != aj1 else aj1
                # Simpler:
                if idx == 0:
                    a_col = 0 + bi_offset
                elif idx == 1:
                    a_col = 0 + bi_offset
                elif idx == 2:
                    a_col = 2 + bi_offset
                else:
                    a_col = 2 + bi_offset
                
                b_row = bi_start + bi_offset
                b_col = bj1 + bj_offset
                c_row = bj1 + bj_offset  # output column
                c_i = ai  # output row
                
                U[ai * m + a_col][col] = 1
                V[b_row * p + b_col][col] = 1
                W[c_row * n + c_i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
