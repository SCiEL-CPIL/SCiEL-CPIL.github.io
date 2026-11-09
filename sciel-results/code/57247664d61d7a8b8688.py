"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # We will use Strassen for the 4x4x4 part (rank 49) and Schoolbook for the 4x4x1 part (rank 16).
    # Total rank = 65.
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    # a_idx: index in U (i*m + j)
    # b_idx: index in V (j*p + k)
    # c_idx: index in W (k*n + i)
    # coeffs: (coeff_a, coeff_b, coeff_c)
    def add_term(a_idx, b_idx, c_idx, ca, cb, cc):
        nonlocal col
        if ca != 0: U[a_idx][col] += ca
        if cb != 0: V[b_idx][col] += cb
        if cc != 0: W[c_idx][col] += cc
        col += 1

    # Strassen's algorithm for 2x2 matrices
    # Inputs: A, B (2x2)
    # Outputs: 7 linear combinations for A and B, and how they contribute to C
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B10+B11)
    # M4 = (A11)(B01-B11)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # We need to apply this to 4x4 matrices by viewing them as 2x2 blocks of 2x2 matrices.
    # Let A be 4x4, split into 4 blocks of 2x2: A00, A01, A10, A11
    # Let B be 4x4 (first 4 cols of B), split into 4 blocks of 2x2: B00, B01, B10, B11
    # Let C be 4x4 (first 4 cols of C), split into 4 blocks of 2x2: C00, C01, C10, C11
    
    # The 7 Strassen products M1..M7 are 2x2 matrix multiplications.
    # Each 2x2 multiplication is decomposed using Strassen again (rank 7).
    
    # We define the linear combinations for the "Strassen layer 1" (4x4 level)
    # These combinations produce 2x2 matrices (blocks).
    # Then we apply "Strassen layer 2" (2x2 level) to each block multiplication.
    
    # To keep it simple and robust, we will generate the terms programmatically.
    
    # Step 1: Define the 7 Strassen terms for the 4x4 level.
    # Each term is (A_blocks, B_blocks) where blocks are indices 0..3 for 2x2 submatrices.
    # Block 0: rows 0-1, cols 0-1
    # Block 1: rows 0-1, cols 2-3
    # Block 2: rows 2-3, cols 0-1
    # Block 3: rows 2-3, cols 2-3
    
    # We will store the coefficients for each 2x2 block element.
    # A element (i,j) is in block (i//2, j//2).
    
    strassen_4x4_terms = [
        # M1: (A00+A11)(B00+B11) -> Blocks 0 and 3
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2: (A10+A11)(B00) -> Blocks 2,3 and 0
        ([(2, 1), (3, 1)], [(0, 1)]),
        # M3: (A00)(B10+B11) -> Block 0 and 2,3
        ([(0, 1)], [(2, 1), (3, 1)]),
        # M4: (A11)(B01-B11) -> Block 3 and 1,3
        ([(3, 1)], [(1, 1), (3, -1)]),
        # M5: (A00+A01)(B11) -> Blocks 0,1 and 3
        ([(0, 1), (1, 1)], [(3, 1)]),
        # M6: (A10-A00)(B00+B01) -> Blocks 2,0 and 0,1
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        # M7: (A01-A11)(B10+B11) -> Blocks 1,3 and 2,3
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)])
    ]
    
    # For each of these 7 terms, we have a 2x2 matrix multiplication:
    # X = sum of A_blocks, Y = sum of B_blocks.
    # X and Y are 2x2 matrices (conceptually, but they are sums of 2x2 blocks, so they are 4x4? No.)
    # Wait. The blocks are 2x2. So A00 is a 2x2 matrix.
    # X = A00 + A11 is a 2x2 matrix? No, A00 and A11 are 2x2 matrices in different positions of the 4x4 matrix.
    # When we say "A00+A11", we mean the 2x2 matrix formed by adding the top-left 2x2 block and the bottom-right 2x2 block?
    # No, Strassen works on the blocks themselves.
    # If A is partitioned into 2x2 blocks, then A00 is a 2x2 matrix.
    # The product M1 = (A00+A11)(B00+B11) is the product of two 2x2 matrices.
    # So X is a 2x2 matrix, Y is a 2x2 matrix.
    # The result is a 2x2 matrix.
    # This 2x2 result contributes to the C blocks.
    
    # So, for each of the 7 terms, we need to:
    # 1. Construct the 2x2 matrix X (coefficients for A elements).
    # 2. Construct the 2x2 matrix Y (coefficients for B elements).
    # 3. Decompose the 2x2 multiplication X*Y into 7 rank-1 terms (using Strassen).
    # 4. Map the resulting 2x2 C contribution to the correct 4x4 C blocks.
    
    # Let's define the mapping from local 2x2 indices to global 4x4 indices.
    # Global A index: i*4 + j.
    # Block 0 (top-left): i in {0,1}, j in {0,1}. Local (r,c) -> Global (r, c).
    # Block 1 (top-right): i in {0,1}, j in {2,3}. Local (r,c) -> Global (r, c+2).
    # Block 2 (bottom-left): i in {2,3}, j in {0,1}. Local (r,c) -> Global (r+2, c).
    # Block 3 (bottom-right): i in {2,3}, j in {2,3}. Local (r,c) -> Global (r+2, c+2).
    
    def get_a_global_index(block, r, c):
        row_offset = (block // 2) * 2
        col_offset = (block % 2) * 2
        return (row_offset + r) * 4 + (col_offset + c)
        
    def get_b_global_index(block, r, c):
        # B is 4x5. We are using the first 4 columns (0-3) for this part.
        # Block 0: rows 0-1, cols 0-1.
        # Block 1: rows 0-1, cols 2-3.
        # Block 2: rows 2-3, cols 0-1.
        # Block 3: rows 2-3, cols 2-3.
        row_offset = (block // 2) * 2
        col_offset = (block % 2) * 2
        return (row_offset + r) * 5 + (col_offset + c)
        
    def get_c_global_index(block, r, c):
        # C is 4x5. First 4 columns.
        # Block 0: rows 0-1, cols 0-1.
        # Block 1: rows 0-1, cols 2-3.
        # Block 2: rows 2-3, cols 0-1.
        # Block 3: rows 2-3, cols 2-3.
        row_offset = (block // 2) * 2
        col_offset = (block % 2) * 2
        i = row_offset + r
        k = col_offset + c
        return k * 4 + i # W index is k*n + i

    # Strassen terms for 2x2 multiplication (to be applied to X and Y)
    # Returns list of (x_idx, y_idx, c_local_idx, coeff)
    # Where x_idx, y_idx are (r,c) in 2x2, c_local_idx is (r,c) in 2x2 result.
    # We use Strassen's 7 products.
    # M1 = (X00+X11)(Y00+Y11) -> C00(+), C11(+)
    # M2 = (X10+X11)(Y00) -> C10(+), C11(-)
    # M3 = (X00)(Y10+Y11) -> C01(+), C11(+)
    # M4 = (X11)(Y01-Y11) -> C00(+), C10(+)
    # M5 = (X00+X01)(Y11) -> C00(-), C01(+)
    # M6 = (X10-X00)(Y00+Y01) -> C11(+)
    # M7 = (X01-X11)(Y10+Y11) -> C00(-)
    
    # This is getting complex to map the linear combinations.
    # It is easier to just expand the 7 Strassen products of the 4x4 level into scalar terms directly.
    
    # Alternative: Just use the known rank 49 decomposition for 4x4x4.
    # I will construct the 49 terms by iterating through the 7 Strassen blocks and 7 Strassen sub-blocks.
    
    # For each of the 7 Strassen terms in the 4x4 level:
    # We have X (2x2) and Y (2x2).
    # We apply Strassen to X*Y.
    # The "elements" of X are linear combinations of A blocks.
    # Specifically, X[r,c] is a sum of A[global_i, global_j].
    # Since A blocks are 2x2, X[r,c] is actually a 2x2 matrix? No.
    # In the 4x4 Strassen, the "elements" are 2x2 matrices.
    # So X is a 2x2 matrix of 2x2 matrices? No, X is a 2x2 matrix where each entry is a 2x2 block of A.
    # Wait, no.
    # A is 4x4. Partitioned into 2x2 blocks.
    # A00 is a 2x2 matrix.
    # X = A00 + A11. This is a 2x2 matrix (sum of two 2x2 matrices).
    # So X is just a 2x2 matrix of scalars? No, A00 contains 4 scalars.
    # So X contains 4 scalars.
    # So X is a 2x2 matrix of scalars.
    # Correct.
    
    # So, for each of the 7 Strassen terms at the 4x4 level:
    # 1. Calculate X (2x2) and Y (2x2) as linear combinations of A and B entries.
    # 2. Apply Strassen to X*Y (2x2 multiplication).
    # 3. This generates 7 products of the form (linear_comb_A) * (linear_comb_B).
    # 4. Each such product contributes to specific entries of the 2x2 result matrix Z.
    # 5. Z is then added to specific 2x2 blocks of C.
    
    # Let's implement this.
    
    for term_idx, (a_blocks, b_blocks) in enumerate(strassen_4x4_terms):
        # Construct X and Y coefficients
        # X[r,c] is a dict of {global_a_idx: coeff}
        X = [[{} for _ in range(2)] for _ in range(2)]
        Y = [[{} for _ in range(2)] for _ in range(2)]
        
        for block, coeff in a_blocks:
            for r in range(2):
                for c in range(2):
                    g_idx = get_a_global_index(block, r, c)
                    X[r][c][g_idx] = X[r][c].get(g_idx, 0) + coeff
                    
        for block, coeff in b_blocks:
            for r in range(2):
                for c in range(2):
                    g_idx = get_b_global_index(block, r, c)
                    Y[r][c][g_idx] = Y[r][c].get(g_idx, 0) + coeff
        
        # Now apply Strassen to X*Y
        # We need the 7 Strassen products for 2x2 matrices.
        # We will iterate through the 7 Strassen terms for the 2x2 level.
        
        # Define the 7 Strassen terms for 2x2 level
        # Each term is (x_indices, y_indices, c_contributions)
        # x_indices: list of (r, c, coeff)
        # y_indices: list of (r, c, coeff)
        # c_contributions: list of (r, c, coeff)
        
        strassen_2x2_terms = [
            # M1: (X00+X11)(Y00+Y11) -> C00(+), C11(+)
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # M2: (X10+X11)(Y00) -> C10(+), C11(-)
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
            # M3: (X00)(Y10+Y11) -> C01(+), C11(+)
            ([(0,0,1)], [(1,0,1), (1,1,1)], [(0,1,1), (1,1,1)]),
            # M4: (X11)(Y01-Y11) -> C00(+), C10(+)
            ([(1,1,1)], [(0,1,1), (1,1,-1)], [(0,0,1), (1,0,1)]),
            # M5: (X00+X01)(Y11) -> C00(-), C01(+)
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
            # M6: (X10-X00)(Y00+Y01) -> C11(+)
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
            # M7: (X01-X11)(Y10+Y11) -> C00(-)
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,-1)])
        ]
        
        for x_terms, y_terms, c_terms in strassen_2x2_terms:
            # Construct linear combination for X_part
            x_comb = {}
            for r, c, coeff in x_terms:
                for g_idx, val in X[r][c].items():
                    x_comb[g_idx] = x_comb.get(g_idx, 0) + val * coeff
                    
            # Construct linear combination for Y_part
            y_comb = {}
            for r, c, coeff in y_terms:
                for g_idx, val in Y[r][c].items():
                    y_comb[g_idx] = y_comb.get(g_idx, 0) + val * coeff
            
            # This product contributes to multiple C entries.
            # We add one term for each non-zero pair in x_comb and y_comb?
            # No, the rank-1 term is (sum x_i A_i) * (sum y_j B_j) * C_k.
            # This is a single rank-1 term in the tensor decomposition if we fix the C index.
            # But here, the same product contributes to multiple C indices.
            # We can write:
            # (sum x_i A_i) (sum y_j B_j) sum_k c_k C_k
            # This is a sum of rank-1 terms: sum_k c_k (sum x_i A_i) (sum y_j B_j) C_k
            # Each k gives a rank-1 term.
            
            for cr, cc, c_coeff in c_terms:
                if c_coeff == 0: continue
                
                # Determine the global C index
                # The result Z is a 2x2 matrix.
                # Z[cr, cc] contributes to which C block?
                # The Strassen term at 4x4 level (term_idx) determines which C blocks are affected.
                # We need to map the local 2x2 result Z to global C blocks.
                
                # Let's re-evaluate the mapping.
                # The 7 Strassen terms at 4x4 level produce M1..M7.
                # C00 (block) = M1 + M4 - M5 - M7
                # C01 (block) = M3 + M5
                # C10 (block) = M2 + M4
                # C11 (block) = M1 - M2 + M3 + M6
                
                # So, for term_idx, we know which C blocks it contributes to and with what sign.
                # But wait, the 2x2 Strassen inside produces a 2x2 matrix Z.
                # This Z IS the contribution to the specific C block(s).
                # No, M1 is a 2x2 matrix. It contributes to C00 and C11.
                # So Z[0,0] contributes to C00[0,0] and C11[0,0]?
                # No. C00 is a 2x2 block. M1 is a 2x2 matrix.
                # C00 = M1 + ...
                # So the entire 2x2 matrix M1 is added to the 2x2 block C00.
                # So Z[r,c] contributes to C_block[r,c].
                
                # So we need to know which C block(s) this term_idx contributes to.
                c_block_contribs = []
                if term_idx == 0: c_block_contribs = [(0, 1), (3, 1)] # C00, C11
                if term_idx == 1: c_block_contribs = [(2, 1), (3, -1)] # C10, C11
                if term_idx == 2: c_block_contribs = [(1, 1), (3, 1)] # C01, C11
                if term_idx == 3: c_block_contribs = [(0, 1), (2, 1)] # C00, C10
                if term_idx == 4: c_block_contribs = [(0, -1), (1, 1)] # C00, C01
                if term_idx == 5: c_block_contribs = [(3, 1)] # C11
                if term_idx == 6: c_block_contribs = [(0, -1)] # C00
                
                for c_block, block_sign in c_block_contribs:
                    # The local index (cr, cc) in Z maps to global index in C block c_block
                    g_c_idx = get_c_global_index(c_block, cr, cc)
                    
                    # Now we add the rank-1 term:
                    # U: x_comb, V: y_comb, W: 1 at g_c_idx
                    # Coefficient is block_sign * c_coeff
                    final_coeff = block_sign * c_coeff
                    
                    # We must add this term to U, V, W.
                    # But U and V are linear combinations.
                    # We add to the current column.
                    
                    # However, we cannot just "add" to the column if we are creating a new rank-1 term.
                    # Yes, we are.
                    
                    # Check if x_comb or y_comb is empty
                    if not x_comb or not y_comb: continue
                    
                    # Add to U
                    for a_idx, val in x_comb.items():
                        U[a_idx][col] += val * final_coeff
                    # Add to V
                    for b_idx, val in y_comb.items():
                        V[b_idx][col] += val * final_coeff
                    # Add to W
                    W[g_c_idx][col] += final_coeff
                    
                    col += 1

    # Now handle the last column (index 4) of B.
    # B is 4x5. Last column is 4x1.
    # A is 4x4.
    # C is 4x1 (last column).
    # We use Schoolbook: 16 terms.
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    
    for i in range(4):
        for j in range(4):
            a_idx = i * 4 + j
            b_idx = j * 5 + 4
            c_idx = 4 * 4 + i # k=4, n=4, i=i -> 4*4+i
            
            U[a_idx][col] = 1
            V[b_idx][col] = 1
            W[c_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
