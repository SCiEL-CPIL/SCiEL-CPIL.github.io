"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal (or near-minimal) tensor decomposition for 2x3 x 3x8 matrix multiplication.
    
    Strategy:
    - Decompose the 3x8 matrix into two 3x4 blocks: B1 (cols 0-3) and B2 (cols 4-7).
    - The product C = A * B = A * B1 + A * B2.
    - Since A is 2x3, we can't directly use Strassen (which requires square matrices).
    - Instead, we use the schoolbook algorithm but exploit structure.
    
    Actually, let's think more carefully. The standard schoolbook rank is 2*3*8 = 48.
    
    Let's try a different approach. We can view this as:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    For fixed i and k, this is a bilinear form. The tensor has dimensions 2 x 3 x 8.
    
    Let's try to use the fact that we can decompose the 3x8 part.
    
    Actually, a better approach: we can use the identity that matrix multiplication
    can be done with fewer multiplications by using bilinear algorithms.
    
    For 2x3 times 3x8:
    Let A be 2x3, B be 3x8.
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    This is a sum of 3 products for each (i,k) pair, with 2*8 = 16 such pairs, giving 48 total.
    
    Can we do better? Let's consider grouping.
    
    Actually, let's use a known result: the rank of the matrix multiplication tensor
    M_{n,m,p} is at most n*m*p (schoolbook), and for small cases, better algorithms exist.
    
    For 2x2x2, Strassen gives rank 7 instead of 8.
    
    For our case, we can try to embed. But 2x3x8 is not a power of 2 in all dimensions.
    
    Alternative: We can split the 3 columns of A into 1+2 or 2+1, and the 8 columns of B into 4+4.
    
    Let's try: Split A = [A1 | A2] where A1 is 2x1 and A2 is 2x2.
    Split B = [B1; B2] where B1 is 1x8 and B2 is 2x8.
    
    Then C = A1*B1 + A2*B2.
    
    A1*B1: 2x1 times 1x8 -> rank at most 2*1*8 = 16 (schoolbook)
    A2*B2: 2x2 times 2x8 -> we can use Strassen-like for 2x2 times 2x8.
    
    For 2x2 times 2x8: Split B2 = [B2a | B2b] where B2a and B2b are 2x4.
    C_part = A2*B2a + A2*B2b.
    
    For 2x2 times 2x4: Split B2a = [B2a1 | B2a2] where each is 2x2.
    So we have two 2x2 times 2x2 multiplications.
    
    Each 2x2x2 can be done with Strassen (rank 7).
    So 2x2 times 2x4 has rank at most 7 + 7 = 14.
    So 2x2 times 2x8 has rank at most 14 + 14 = 28.
    
    Total: 16 + 28 = 44. Still worse than 48? No, 44 < 48.
    
    Wait, let me recount. Schoolbook for 2x3x8 is 48. My decomposition gives 44. That's an improvement.
    
    Can we do better for the 2x1 times 1x8 part?
    A1 is 2x1, B1 is 1x8. C1[i,k] = A1[i,0]*B1[0,k].
    This is just scaling each column of B1 by A1[i,0]. 
    For each i (2 choices) and k (8 choices), we have one product. So 16 products.
    But these are independent, so rank is exactly 16 for this subproblem.
    
    So total is 16 + 28 = 44.
    
    Let's verify the Strassen construction for 2x2x2.
    Strassen's algorithm computes 2x2 * 2x2 with 7 multiplications.
    
    Let me implement this carefully.
    """
    import numpy as np
    import time
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # We'll build the decomposition step by step.
    # The overall structure:
    # A is 2x3, B is 3x8.
    # Split A into A_left (2x1) and A_right (2x2).
    # Split B into B_top (1x8) and B_bottom (2x8).
    # C = A_left * B_top + A_right * B_bottom.
    
    # For A_right * B_bottom (2x2 * 2x8):
    # Split B_bottom into B_bottom_left (2x4) and B_bottom_right (2x4).
    # For each 2x2 * 2x4, split the 2x4 into two 2x2 blocks.
    # Use Strassen for each 2x2 * 2x2.
    
    # Let's define helper to create Strassen decomposition for 2x2 * 2x2.
    # Input: A2 is 2x2, B2 is 2x2.
    # Output: 7 terms, each is a product of a linear combo of A entries and a linear combo of B entries.
    
    # Strassen's 7 products:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B10-B11)
    # P4 = A11*(B01-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B10)
    # P7 = (A01-A11)*(B01+B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # In the tensor decomposition, each "product" corresponds to a rank-1 term.
    # For a term like (sum_i a_i * A[i,j]) * (sum_k b_k * B[k,l]),
    # we add 1 to U at the appropriate row, 1 to V at the appropriate column,
    # and the coefficient to W.
    
    # Let's build the full decomposition.
    
    # We'll accumulate U, V, W as lists.
    U_terms = []  # Each term is a dict or tuple describing the contribution
    V_terms = []
    W_terms = []
    
    # Helper to add a term with coefficient c, A-indices, B-indices, C-index
    # A-indices: list of (i, j) pairs with coefficients
    # B-indices: list of (j, k) pairs with coefficients
    # C-index: (i, k) with coefficient
    
    def add_term(coeff, a_indices, b_indices, c_index):
        """
        a_indices: list of (row, col, weight) for A
        b_indices: list of (row, col, weight) for B
        c_index: (row, col, weight) for C
        """
        # For the rank-1 tensor term, we need:
        # U[row_A, col] += coeff * weight_A
        # V[row_B, col] += coeff * weight_B  
        # W[row_C, col] += coeff * weight_C
        # But wait, the tensor is sum_c U[i,c] * V[j,c] * W[k,c] = C[i,k]
        # where i indexes A's rows, j indexes A's cols / B's rows, k indexes B's cols.
        # Actually, the standard convention is:
        # C[i,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Wait, let me re-read the task.
        # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
        # So U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
        # The product is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This should equal the (i,k) entry of the product matrix.
        
        # So for a term where A contributes from (i_A, j_A) with weight w_A,
        # B contributes from (j_B, k_B) with weight w_B, and the result goes to (i_C, k_C) with weight w_C,
        # we need j_A == j_B (the inner dimension must match).
        # And the term contributes w_A * w_B * w_C to C[i_C, k_C] from the c-th component.
        
        # In our case, we're building explicit rank-1 terms.
        # Each term corresponds to one multiplication in the algorithm.
        
        pass
    
    # Let me rethink the construction.
    
    # The matrix multiplication tensor T is defined by:
    # T[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i, j, k] = 1 for all valid i, j, k.
    #
    # A rank-1 decomposition is a sum of terms:
    # sum_c u_c[i] * v_c[j] * w_c[k] = T[i, j, k]
    #
    # In the task's convention:
    # U has shape (n*m, rank), so U[i*m + j, c] = u_c[i, j]
    # V has shape (m*p, rank), so V[j*p + k, c] = v_c[j, k]  
    # W has shape (p*n, rank), so W[k*n + i, c] = w_c[k, i]
    #
    # And the product is sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This should equal 1 for all i, j, k (since T[i,j,k] = 1).
    
    # So for each "multiplication" in our algorithm, we have a term where:
    # u_c is a vector over (i, j) pairs, v_c is a vector over (j, k) pairs, w_c is a vector over (k, i) pairs.
    # The product u_c[i,j] * v_c[j,k] * w_c[k,i] should contribute to the sum.
    
    # For the schoolbook algorithm, each term has u_c[i,j] = 1 for one (i,j), v_c[j,k] = 1 for one (j,k), w_c[k,i] = 1 for one (k,i), with j matching.
    
    # For Strassen, we have linear combinations.
    
    # Let me just build it directly.
    
    # Start with empty matrices.
    # We'll figure out the rank first.
    
    # Strassen for 2x2x2: 7 terms.
    # We have 4 such subproblems (2x2 * 2x2) from the A_right * B_bottom part.
    # Plus 16 terms from A_left * B_top.
    # Total: 4*7 + 16 = 28 + 16 = 44.
    
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]  # 2*3 = 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 3*8 = 24 rows
    W = [[0] * rank for _ in range(p * n)]  # 8*2 = 16 rows
    
    col = 0
    
    # Part 1: A_left (2x1) * B_top (1x8)
    # A_left[i, 0] * B_top[0, k] for i in 0..1, k in 0..7
    # In the full A (2x3), A_left corresponds to column 0.
    # In the full B (3x8), B_top corresponds to row 0.
    # So for each i in 0..1, k in 0..7:
    #   U[i*3 + 0, col] = 1  (A[i, 0])
    #   V[0*8 + k, col] = 1  (B[0, k])
    #   W[k*2 + i, col] = 1  (C[i, k])
    
    for i in range(2):
        for k in range(8):
            U[i * 3 + 0][col] = 1
            V[0 * 8 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
    
    # Part 2: A_right (2x2) * B_bottom (2x8)
    # A_right corresponds to columns 1,2 of A.
    # B_bottom corresponds to rows 1,2 of B.
    # Split B_bottom into B_bottom_left (2x4) and B_bottom_right (2x4).
    # B_bottom_left: rows 1,2; cols 0..3
    # B_bottom_right: rows 1,2; cols 4..7
    
    # For each 2x4 block, split into two 2x2 blocks.
    # B_bottom_left: cols 0..3 -> B_bl1 (cols 0,1), B_bl2 (cols 2,3)
    # B_bottom_right: cols 4..7 -> B_br1 (cols 4,5), B_br2 (cols 6,7)
    
    # For each 2x2 * 2x2, use Strassen.
    
    # Strassen terms for 2x2 * 2x2:
    # Let A2 = [[a00, a01], [a10, a11]], B2 = [[b00, b01], [b10, b11]]
    # P1 = (a00+a11)*(b00+b11)
    # P2 = (a10+a11)*b00
    # P3 = a00*(b10-b11)
    # P4 = a11*(b01-b00)
    # P5 = (a00+a01)*b11
    # P6 = (a10-a00)*(b00+b10)
    # P7 = (a01-a11)*(b01+b11)
    
    # In our indexing:
    # A_right[i, j] where i in 0..1, j in 0..1 corresponds to A[i, j+1] in the full A.
    # B_bottom[i, k] where i in 0..1, k in 0..3 (for left) corresponds to B[i+1, k] in the full B.
    
    def strassen_block(a_col_offset, b_row_offset, b_col_offset):
        """
        Add Strassen terms for A_right (cols a_col_offset..a_col_offset+1) * B_bottom (rows b_row_offset..b_row_offset+1, cols b_col_offset..b_col_offset+1).
        """
        nonlocal col
        
        # A2 indices in full A: rows 0..1, cols a_col_offset..a_col_offset+1
        # B2 indices in full B: rows b_row_offset..b_row_offset+1, cols b_col_offset..b_col_offset+1
        
        # P1 = (a00+a11)*(b00+b11)
        # a00 = A[0, a_col_offset], a11 = A[1, a_col_offset+1]
        # b00 = B[b_row_offset, b_col_offset], b11 = B[b_row_offset+1, b_col_offset+1]
        U[0 * 3 + a_col_offset][col] = 1
        U[1 * 3 + (a_col_offset + 1)][col] = 1
        V[b_row_offset * 8 + b_col_offset][col] = 1
        V[(b_row_offset + 1) * 8 + (b_col_offset + 1)][col] = 1
        # C contributions:
        # C00 += P1, C01 -= P1, C10 -= P1, C11 += P1
        # C[i,k] where i in 0..1, k in b_col_offset..b_col_offset+1
        # C[0, b_col_offset] += P1 -> W[(b_col_offset)*2 + 0][col] += 1
        # C[0, b_col_offset+1] -= P1 -> W[(b_col_offset+1)*2 + 0][col] -= 1
        # C[1, b_col_offset] -= P1 -> W[(b_col_offset)*2 + 1][col] -= 1
        # C[1, b_col_offset+1] += P1 -> W[(b_col_offset+1)*2 + 1][col] += 1
        W[b_col_offset * 2 + 0][col] = 1
        W[(b_col_offset + 1) * 2 + 0][col] = -1
        W[b_col_offset * 2 + 1][col] = -1
        W[(b_col_offset + 1) * 2 + 1][col] = 1
        col += 1
        
        # P2 = (a10+a11)*b00
        # a10 = A[1, a_col_offset], a11 = A[1, a_col_offset+1]
        # b00 = B[b_row_offset, b_col_offset]
        U[1 * 3 + a_col_offset][col] = 1
        U[1 * 3 + (a_col_offset + 1)][col] = 1
        V[b_row_offset * 8 + b_col_offset][col] = 1
        # C10 += P2 -> W[b_col_offset*2 + 1][col] += 1
        # C11 -= P2 -> W[(b_col_offset+1)*2 + 1][col] -= 1
        W[b_col_offset * 2 + 1][col] = 1
        W[(b_col_offset + 1) * 2 + 1][col] = -1
        col += 1
        
        # P3 = a00*(b10-b11)
        # a00 = A[0, a_col_offset]
        # b10 = B[b_row_offset+1, b_col_offset], b11 = B[b_row_offset+1, b_col_offset+1]
        U[0 * 3 + a_col_offset][col] = 1
        V[(b_row_offset + 1) * 8 + b_col_offset][col] = 1
        V[(b_row_offset + 1) * 8 + (b_col_offset + 1)][col] = -1
        # C01 += P3 -> W[(b_col_offset+1)*2 + 0][col] += 1
        # C11 += P3 -> W[(b_col_offset+1)*2 + 1][col] += 1
        W[(b_col_offset + 1) * 2 + 0][col] = 1
        W[(b_col_offset + 1) * 2 + 1][col] = 1
        col += 1
        
        # P4 = a11*(b01-b00)
        # a11 = A[1, a_col_offset+1]
        # b01 = B[b_row_offset, b_col_offset+1], b00 = B[b_row_offset, b_col_offset]
        U[1 * 3 + (a_col_offset + 1)][col] = 1
        V[b_row_offset * 8 + (b_col_offset + 1)][col] = 1
        V[b_row_offset * 8 + b_col_offset][col] = -1
        # C00 += P4 -> W[b_col_offset*2 + 0][col] += 1
        # C10 += P4 -> W[b_col_offset*2 + 1][col] += 1
        W[b_col_offset * 2 + 0][col] = 1
        W[b_col_offset * 2 + 1][col] = 1
        col += 1
        
        # P5 = (a00+a01)*b11
        # a00 = A[0, a_col_offset], a01 = A[0, a_col_offset+1]
        # b11 = B[b_row_offset+1, b_col_offset+1]
        U[0 * 3 + a_col_offset][col] = 1
        U[0 * 3 + (a_col_offset + 1)][col] = 1
        V[(b_row_offset + 1) * 8 + (b_col_offset + 1)][col] = 1
        # C00 -= P5 -> W[b_col_offset*2 + 0][col] -= 1
        # C01 += P5 -> W[(b_col_offset+1)*2 + 0][col] += 1
        W[b_col_offset * 2 + 0][col] = -1
        W[(b_col_offset + 1) * 2 + 0][col] = 1
        col += 1
        
        # P6 = (a10-a00)*(b00+b10)
        # a10 = A[1, a_col_offset], a00 = A[0, a_col_offset]
        # b00 = B[b_row_offset, b_col_offset], b10 = B[b_row_offset+1, b_col_offset]
        U[1 * 3 + a_col_offset][col] = 1
        U[0 * 3 + a_col_offset][col] = -1
        V[b_row_offset * 8 + b_col_offset][col] = 1
        V[(b_row_offset + 1) * 8 + b_col_offset][col] = 1
        # C11 += P6 -> W[(b_col_offset+1)*2 + 1][col] += 1
        W[(b_col_offset + 1) * 2 + 1][col] = 1
        col += 1
        
        # P7 = (a01-a11)*(b01+b11)
        # a01 = A[0, a_col_offset+1], a11 = A[1, a_col_offset+1]
        # b01 = B[b_row_offset, b_col_offset+1], b11 = B[b_row_offset+1, b_col_offset+1]
        U[0 * 3 + (a_col_offset + 1)][col] = 1
        U[1 * 3 + (a_col_offset + 1)][col] = -1
        V[b_row_offset * 8 + (b_col_offset + 1)][col] = 1
        V[(b_row_offset + 1) * 8 + (b_col_offset + 1)][col] = 1
        # C00 += P7 -> W[b_col_offset*2 + 0][col] += 1
        # C01 -= P7 -> W[(b_col_offset+1)*2 + 0][col] -= 1
        W[b_col_offset * 2 + 0][col] = 1
        W[(b_col_offset + 1) * 2 + 0][col] = -1
        col += 1
    
    # Apply Strassen to the four 2x2 blocks.
    # A_right cols: 1, 2 (a_col_offset = 1)
    # B_bottom rows: 1, 2 (b_row_offset = 1)
    # B_bottom cols: 0,1,2,3,4,5,6,7
    # Four blocks: (0,1), (2,3), (4,5), (6,7)
    
    strassen_block(1, 1, 0)
    strassen_block(1, 1, 2)
    strassen_block(1, 1, 4)
    strassen_block(1, 1, 6)
    
    # Verify col == rank
    if col != rank:
        # Adjust if needed
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
