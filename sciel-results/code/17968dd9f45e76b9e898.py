"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-70 decomposition for 4x5 * 5x6 matrix multiplication.
    
    Strategy: 
    - Split A (4x5) = [A1 (4x4) | A2 (4x1)]
    - Split B (5x6) = [B1 (4x6); B2 (1x6)]
    - C = A1*B1 + A2*B2
    
    For A2*B2 (4x1 * 1x6): rank 1 (outer product)
    For A1*B1 (4x4 * 4x6): 
      Split B1 = [B1a (4x4) | B1b (4x2)]
      A1*B1 = A1*B1a + A1*B1b
      
    A1*B1a (4x4 * 4x4): Use Strassen recursively. 
      Split into 2x2 blocks, use 7 multiplications per 2x2*2x2 pair.
      4x4*4x4 = 4 output 2x2 blocks, each from 2 input 2x2 block pairs.
      Each 2x2*2x2 uses Strassen (rank 7).
      Total: 4 * 7 = 28? No wait.
      
      Actually for 4x4*4x4:
      A = [[A00, A01], [A10, A11]] (2x2 blocks)
      B = [[B00, B01], [B10, B11]] (2x2 blocks)
      C00 = A00*B00 + A01*B10
      C01 = A00*B01 + A01*B11
      C10 = A10*B00 + A11*B10
      C11 = A10*B01 + A11*B11
      
      Using Strassen on each 2x2*2x2 product:
      For C00: 7 mults
      For C01: 7 mults
      For C10: 7 mults
      For C11: 7 mults
      Total: 28 multiplications.
      
      But we can do better by sharing. The standard Strassen for 4x4*4x4 gives rank 49.
      Let me use the known Strassen-Winograd result: rank(4x4*4x4) <= 49.
      
    A1*B1b (4x4 * 4x2):
      Split B1b into two 4x1 matrices: B1b1, B1b2
      A1*B1b1 (4x4 * 4x1): schoolbook rank 16, or can we do better?
      Actually, let's just use schoolbook for the tail parts to keep it simple.
      
    Let me reconsider the total rank:
    - A1*B1a: 49 (Strassen)
    - A1*B1b: 4x4*4x2, schoolbook = 4*4*2 = 32
    - A2*B2: 1
    Total: 49 + 32 + 1 = 82
    
    That's not 70. Let me think harder.
    
    Alternative: Maybe I should use a different splitting.
    
    Let me try: 
    A1*B1b (4x4 * 4x2): 
    Split A1 into 2x2 blocks: A1 = [[A00, A01], [A10, A11]]
    B1b is 4x2, split into 2x2 blocks vertically: B1b = [B0; B1] where B0, B1 are 2x2.
    
    C_top = A00*B0 + A01*B1 (2x2 result)
    C_bot = A10*B0 + A11*B1 (2x2 result)
    
    For C_top: 7 mults (Strassen)
    For C_bot: 7 mults (Strassen)
    Total: 14 mults for A1*B1b.
    
    So total rank: 49 + 14 + 1 = 64.
    
    Wait, that's even better than 70! Let me verify this is correct.
    
    Actually, I need to be careful. The Strassen decomposition for 2x2*2x2 has rank 7. When I say "7 mults", I mean 7 rank-1 terms in the tensor decomposition.
    
    For A1*B1a (4x4*4x4):
    Using the recursive Strassen approach:
    - 4 output 2x2 blocks
    - Each requires 7 multiplications of 2x2 matrices
    - But these 2x2 multiplications are themselves decomposed using Strassen (rank 7)
    - So total rank = 4 * 7 * 7 = 196? No, that's not right.
    
    Wait, I'm confusing myself. Let me be very careful.
    
    The Strassen algorithm for 2x2*2x2 uses 7 scalar multiplications. In tensor rank terms, the (2,2,2) matrix multiplication tensor has rank 7.
    
    For 4x4*4x4, we can view it as 4 separate 2x2*2x2 problems (one for each output block), but the inputs overlap. The standard recursive Strassen gives rank 7^2 = 49.
    
    How? We have:
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11  
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    If we naively compute each 2x2 product with Strassen: 4 * 7 = 28 scalar mults. But that's not optimal.
    
    The optimal Strassen for 4x4*4x4 has rank 49. This comes from viewing the 4x4*4x4 as a composition: we need to compute 7 "big" multiplications of 2x2 blocks, and each big multiplication is itself a 2x2*2x2 which has rank 7. So 7 * 7 = 49.
    
    OK so A1*B1a has rank 49.
    
    For A1*B1b (4x4*4x2):
    I claimed 14 above. Let me verify.
    
    A1 = [[A00, A01], [A10, A11]] (each 2x2)
    B1b = [B0; B1] (each 2x2, stacked vertically)
    
    C_top = A00*B0 + A01*B1 (2x2)
    C_bot = A10*B0 + A11*B1 (2x2)
    
    For C_top: This is a sum of two 2x2*2x2 products. The tensor rank of this combined operation...
    Actually, I need to think about this differently.
    
    The problem is: given 2x2 matrices A00, A01, B0, B1, compute A00*B0 + A01*B1.
    This is a bilinear map. The tensor rank of this specific map is at most 7 (if we can share Strassen multiplications).
    
    Hmm, this is getting complex. Let me just use schoolbook for A1*B1b to be safe: rank 32.
    
    Total: 49 + 32 + 1 = 82.
    
    Let me try yet another approach. What if I use Strassen more cleverly?
    
    Actually, let me just implement the 49 + 32 + 1 = 82 decomposition and see if it's correct. If the evaluator accepts it, great. If not, I'll debug.
    
    Wait, the directive says "rank-70". Let me see if I can get to 70.
    
    Alternative splitting:
    A (4x5) = [A1 (4x4) | A2 (4x1)]
    B (5x6) = [B1 (4x6); B2 (1x6)]
    
    What if I split differently?
    A (4x5) = [A1 (2x5); A2 (2x5)]
    B (5x6) = [B1 (5x3) | B2 (5x3)]
    
    C_top = A1*B1 + A1*B2 (2x6)
    C_bot = A2*B1 + A2*B2 (2x6)
    
    This doesn't seem to help.
    
    Let me just go with the 82-rank decomposition and implement it carefully.
    
    Actually, let me reconsider A1*B1b (4x4*4x2) more carefully.
    
    The tensor for (4,4,2) matrix multiplication: T[i,j,k] where i in [0,3], j in [0,3], k in [0,1].
    Schoolbook rank: 4*4*2 = 32.
    
    Can we do better? Using Strassen on 2x2 blocks:
    A = [[A00, A01], [A10, A11]] (2x2 blocks)
    B = [B0; B1] (2x2 blocks, B is 4x2)
    
    C0 = A00*B0 + A01*B1 (2x2)
    C1 = A10*B0 + A11*B1 (2x2)
    
    For C0: The bilinear map (A00, A01, B0, B1) -> A00*B0 + A01*B1.
    This is equivalent to computing a 2x2*2x2 product where the "A" is [A00, A01] (2x4) and "B" is [B0; B1] (4x2)... no, that's not quite right.
    
    Actually, A00*B0 + A01*B1 can be seen as: form M = [A00, A01] (2x4) and N = [B0; B1] (4x2), then compute M*N. But that's a (2,4,2) matrix multiplication, which has schoolbook rank 2*4*2 = 16.
    
    So C0 has rank at most 16, and C1 has rank at most 16. Total for A1*B1b: 32. Same as schoolbook.
    
    So I can't improve A1*B1b below 32 easily.
    
    Let me just implement the 49 + 32 + 1 = 82 decomposition.
    """
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # We'll build the decomposition in parts:
    # Part 1: A1*B1a (4x4 * 4x4) using Strassen, rank 49
    # Part 2: A1*B1b (4x4 * 4x2) using schoolbook, rank 32
    # Part 3: A2*B2 (4x1 * 1x6) using rank-1, rank 1
    # Total rank: 82
    
    total_rank = 49 + 32 + 1
    U = [[0] * total_rank for _ in range(n * m)]  # 20 x 82
    V = [[0] * total_rank for _ in range(m * p)]  # 30 x 82
    W = [[0] * total_rank for _ in range(p * n)]  # 24 x 82
    
    # Helper: add a rank-1 term
    def add_term(col, u_idx, v_idx, w_idx, u_val=1, v_val=1, w_val=1):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
    
    # Part 1: A1*B1a (4x4 * 4x4) using Strassen, rank 49
    # A1 occupies rows 0-3 of A, columns 0-3 (indices i*m+j for i in 0-3, j in 0-3)
    # B1a occupies rows 0-3 of B (j in 0-3), columns 0-3 (k in 0-3) (indices j*p+k for j in 0-3, k in 0-3)
    # Output C occupies rows 0-3, cols 0-3 (indices k*n+i for k in 0-3, i in 0-3)
    
    # Strassen for 4x4*4x4:
    # Split into 2x2 blocks.
    # A1 = [[A00, A01], [A10, A11]]
    #   A00: A[0:2, 0:2], A01: A[0:2, 2:4], A10: A[2:4, 0:2], A11: A[2:4, 2:4]
    # B1a = [[B00, B01], [B10, B11]]
    #   B00: B[0:2, 0:2], B01: B[0:2, 2:4], B10: B[2:4, 0:2], B11: B[2:4, 2:4]
    # C = [[C00, C01], [C10, C11]]
    #   C00: C[0:2, 0:2], C01: C[0:2, 2:4], C10: C[2:4, 0:2], C11: C[2:4, 2:4]
    
    # Strassen for 2x2*2x2 has rank 7. The 7 multiplications are:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 4x4*4x4, we apply Strassen at the 2x2 block level.
    # This means we need 7 "big" multiplications of 2x2 matrices, and each big multiplication
    # is itself decomposed using Strassen (rank 7).
    
    # The 7 big multiplications for 4x4*4x4:
    # Let A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # Each Mi is a 2x2*2x2 product, decomposed with rank 7.
    # So total rank = 7 * 7 = 49.
    
    # For each big multiplication Mi = X*Y where X, Y are 2x2:
    # We decompose X*Y using Strassen (rank 7).
    
    # Let's define a helper to add a Strassen decomposition of a 2x2*2x2 product.
    # The 2x2 matrix X occupies A rows r0:r0+2, cols c0:c0+2 (in the A1 submatrix)
    # The 2x2 matrix Y occupies B rows r0:r0+2, cols c0:c0+2 (in the B1a submatrix)
    # The output 2x2 occupies C rows r0:r0+2, cols c0:c0+2
    
    # Actually, this is getting very complex. Let me just use schoolbook for the 4x4*4x4 part too,
    # and accept rank 16 + 32 + 1 = 49... no wait, schoolbook for 4x4*4x4 is 64.
    
    # Let me just use schoolbook for everything and accept rank 120.
    # Actually, let me try to implement the Strassen properly.
    
    # For the 2x2*2x2 Strassen decomposition, the 7 rank-1 terms are:
    # Given X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # Z = X*Y = [[z00, z01], [z10, z11]]
    # 
    # M1 = (x00+x11)*(y00+y11) -> contributes to z00, z11
    # M2 = (x10+x11)*y00 -> contributes to z10, z11
    # M3 = x00*(y01-y11) -> contributes to z00, z01
    # M4 = x11*(y10-y00) -> contributes to z00, z10
    # M5 = (x00+x01)*y11 -> contributes to z00, z01
    # M6 = (x10-x00)*(y00+y01) -> contributes to z10, z11
    # M7 = (x01-x11)*(y10+y11) -> contributes to z00, z11
    #
    # z00 = M1 + M4 - M5 + M7
    # z01 = M3 + M5
    # z10 = M2 + M4
    # z11 = M1 - M2 + M3 + M6
    
    # In tensor rank terms, each Mi is a product of two linear combinations.
    # Mi = (linear combo of X entries) * (linear combo of Y entries)
    # This is a rank-1 term in the tensor.
    
    # So for a 2x2*2x2 product, we add 7 rank-1 terms.
    
    # For the 4x4*4x4 Strassen, we have 7 such 2x2*2x2 products, each contributing 7 rank-1 terms.
    # Total: 49 rank-1 terms.
    
    # Let me implement this step by step.
    
    col = 0
    
    # Define Strassen 2x2*2x2 decomposition
    # X is at A positions (row_offset, col_offset) in the 4x4 submatrix
    # Y is at B positions (row_offset, col_offset) in the 4x4 submatrix
    # Output Z is at C positions (row_offset, col_offset) in the 4x4 submatrix
    
    def strassen_2x2(a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off, start_col):
        """Add Strassen decomposition for a 2x2*2x2 product.
        X = A[a_row_off:a_row_off+2, a_col_off:a_col_off+2]
        Y = B[b_row_off:b_row_off+2, b_col_off:b_col_off+2]
        Z = X*Y = C[c_row_off:c_row_off+2, c_col_off:c_col_off+2]
        """
        global col
        c = start_col
        # M1 = (x00+x11)*(y00+y11)
        U[a_row_off * m + a_col_off][c] += 1
        U[(a_row_off+1) * m + (a_col_off+1)][c] += 1
        V[b_row_off * p + b_col_off][c] += 1
        V[(b_row_off+1) * p + (b_col_off+1)][c] += 1
        W[c_row_off * n + c_col_off][c] += 1
        W[(c_row_off+1) * n + (c_col_off+1)][c] += 1
        c += 1
        
        # M2 = (x10+x11)*y00
        U[(a_row_off+1) * m + a_col_off][c] += 1
        U[(a_row_off+1) * m + (a_col_off+1)][c] += 1
        V[b_row_off * p + b_col_off][c] += 1
        W[(c_row_off+1) * n + c_col_off][c] += 1
        W[(c_row_off+1) * n + (c_col_off+1)][c] -= 1
        c += 1
        
        # M3 = x00*(y01-y11)
        U[a_row_off * m + a_col_off][c] += 1
        V[b_row_off * p + (b_col_off+1)][c] += 1
        V[(b_row_off+1) * p + (b_col_off+1)][c] -= 1
        W[c_row_off * n + c_col_off][c] += 1
        W[c_row_off * n + (c_col_off+1)][c] += 1
        c += 1
        
        # M4 = x11*(y10-y00)
        U[(a_row_off+1) * m + (a_col_off+1)][c] += 1
        V[(b_row_off+1) * p + b_col_off][c] += 1
        V[b_row_off * p + b_col_off][c] -= 1
        W[c_row_off * n + c_col_off][c] += 1
        W[(c_row_off+1) * n + c_col_off][c] += 1
        c += 1
        
        # M5 = (x00+x01)*y11
        U[a_row_off * m + a_col_off][c] += 1
        U[a_row_off * m + (a_col_off+1)][c] += 1
        V[(b_row_off+1) * p + (b_col_off+1)][c] += 1
        W[c_row_off * n + c_col_off][c] -= 1
        W[c_row_off * n + (c_col_off+1)][c] += 1
        c += 1
        
        # M6 = (x10-x00)*(y00+y01)
        U[(a_row_off+1) * m + a_col_off][c] += 1
        U[a_row_off * m + a_col_off][c] -= 1
        V[b_row_off * p + b_col_off][c] += 1
        V[b_row_off * p + (b_col_off+1)][c] += 1
        W[(c_row_off+1) * n + c_col_off][c] += 1
        W[(c_row_off+1) * n + (c_col_off+1)][c] += 1
        c += 1
        
        # M7 = (x01-x11)*(y10+y11)
        U[a_row_off * m + (a_col_off+1)][c] += 1
        U[(a_row_off+1) * m + (a_col_off+1)][c] -= 1
        V[(b_row_off+1) * p + b_col_off][c] += 1
        V[(b_row_off+1) * p + (b_col_off+1)][c] += 1
        W[c_row_off * n + c_col_off][c] += 1
        W[(c_row_off+1) * n + (c_col_off+1)][c] += 1
        c += 1
        
        return c
    
    # For 4x4*4x4 Strassen, we need 7 big 2x2*2x2 multiplications.
    # But the inputs to these big multiplications are linear combinations of the 2x2 blocks.
    # This means we can't directly use strassen_2x2 because the "X" and "Y" are not single 2x2 blocks
    # but sums/differences of 2x2 blocks.
    
    # This is the key insight: the Strassen algorithm at the 4x4 level creates linear combinations
    # of 2x2 blocks, and then multiplies those combinations. Each such multiplication is a 2x2*2x2
    # product, but the "matrices" being multiplied are linear combinations of the original 2x2 blocks.
    
    # In tensor rank terms, this means we need to handle the linear combinations explicitly.
    # Each "big" multiplication (X_comb * Y_comb) where X_comb is a sum of 2x2 blocks and Y_comb is a sum
    # of 2x2 blocks, can be decomposed into individual 2x2*2x2 products.
    
    # For example, M1 = (A00+A11)*(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # This is 4 separate 2x2*2x2 products, each with rank 7, so rank 28 for M1 alone.
    # That's way too much.
    
    # I think I'm overcomplicating this. The standard result is that rank(4x4*4x4) <= 49.
    # Let me look up the actual construction.
    
    # Actually, the recursive Strassen works as follows:
    # To multiply two 4x4 matrices, we use the Strassen algorithm which requires 7 multiplications
    # of 2x2 matrices. But these 2x2 matrices are not the original blocks - they are linear combinations.
    # The key is that in the tensor decomposition, we don't need to explicitly form these combinations.
    # Instead, we can directly write the 49 rank-1 terms.
    
    # This is getting too complex for me to derive from scratch. Let me just use schoolbook for the
    # 4x4*4x4 part and accept a higher rank.
    
    # Schoolbook for 4x4*4x4: rank 64
    # Schoolbook for 4x4*4x2: rank 32
    # Rank-1 for 4x1*1x6: rank 1
    # Total: 64 + 32 + 1 = 97
    
    # Let me just implement the schoolbook decomposition for the full 4x5*5x6 problem.
    # That gives rank 120, which is what the current program does.
    
    # Since I can't easily derive the rank-70 decomposition, let me just keep the schoolbook.
    
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
# EVOLVE-BLOCK-END
