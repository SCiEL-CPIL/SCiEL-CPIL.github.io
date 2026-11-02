import numpy as np

def construct():
    n, m, p = 4, 5, 5
    # We will build the rank 85 decomposition
    # 49 from 4x4x4 Strassen
    # 16 from 4x4x1 Schoolbook
    # 20 from 4x1x5 Schoolbook
    
    terms = []
    
    # Helper to add a term to the list
    # u_idx, v_idx, w_idx are indices into U, V, W rows
    # coeff is the scalar multiplier
    def add_term(u_idx, v_idx, w_idx, coeff):
        if coeff != 0:
            terms.append((u_idx, v_idx, w_idx, coeff))
            
    # --- Part 1: 4x4x4 Strassen ---
    # We apply Strassen to 2x2 blocks of 2x2 matrices.
    # A is 4x4, B is 4x4, C is 4x4.
    # Strassen generates 7 terms for 2x2.
    # Each term is a product of two 2x2 matrices.
    # We expand each 2x2 product into 7 scalar products (Strassen again? No, 2x2x2 schoolbook is 4, Strassen is 7).
    # Wait, the rank of 2x2x2 is 7.
    # So 4x4x4 rank is 7*7=49.
    # We need the 7 terms for the outer Strassen, and for each term, we need the 7 terms for the inner 2x2 Strassen.
    
    # Outer Strassen terms (7 of them)
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # Inner Strassen terms (7 of them) for a 2x2 product P = X Y
    # X, Y are 2x2.
    # Inner M1 = (X00 + X11)(Y00 + Y11)
    # Inner M2 = (X10 + X11)Y00
    # Inner M3 = X00(Y01 - Y11)
    # Inner M4 = X11(Y10 - Y00)
    # Inner M5 = (X00 + X01)Y11
    # Inner M6 = (X10 - X00)(Y00 + Y01)
    # Inner M7 = (X01 - X11)(Y10 + Y11)
    
    # We need to map the scalar products to U, V, W indices.
    # For a scalar product A[i,j] * B[j,k] contributing to C[i,k]:
    # U_idx = i*m + j
    # V_idx = j*p + k
    # W_idx = k*n + i
    
    # We will generate the 49 terms by iterating over the 7 outer terms and 7 inner terms.
    
    # Outer term definitions:
    # (u_coeffs, v_coeffs) where coeffs are for A and B blocks.
    # A blocks: A00, A01, A10, A11 (2x2 each)
    # B blocks: B00, B01, B10, B11 (2x2 each)
    
    # It is easier to just expand the 49 terms explicitly using the scalar indices.
    # For each of the 49 terms, we have a linear combination of scalar products.
    
    # Let's define the 7 outer terms as lists of (A_block, coeff, B_block, coeff)
    # A_block is (row_offset, col_offset) for the 2x2 block.
    # B_block is (row_offset, col_offset) for the 2x2 block.
    
    outer_terms = [
        # M1: (A00+A11)(B00+B11)
        [((0,0), 1), ((0,0), 1)], # A00 B00
        [((0,0), 1), ((0,1), 1)], # A00 B11
        [((1,0), 1), ((0,0), 1)], # A11 B00
        [((1,0), 1), ((0,1), 1)], # A11 B11
        # M2: (A10+A11)B00
        [((1,1), 1), ((0,0), 1)], # A10 B00
        [((1,0), 1), ((0,0), 1)], # A11 B00
        # M3: A00(B01-B11)
        [((0,0), 1), ((0,1), 1)], # A00 B01
        [((0,0), 1), ((1,1), -1)], # A00 -B11
        # M4: A11(B10-B00)
        [((1,0), 1), ((1,0), 1)], # A11 B10
        [((1,0), 1), ((0,0), -1)], # A11 -B00
        # M5: (A00+A01)B11
        [((0,0), 1), ((0,1), 1)], # A00 B11
        [((0,1), 1), ((0,1), 1)], # A01 B11
        # M6: (A10-A00)(B00+B01)
        [((1,1), 1), ((0,0), 1)], # A10 B00
        [((1,1), 1), ((0,1), 1)], # A10 B01
        [((0,0), -1), ((0,0), 1)], # -A00 B00
        [((0,0), -1), ((0,1), 1)], # -A00 B01
        # M7: (A01-A11)(B10+B11)
        [((0,1), 1), ((1,0), 1)], # A01 B10
        [((0,1), 1), ((1,1), 1)], # A01 B11
        [((1,0), -1), ((1,0), 1)], # -A11 B10
        [((1,0), -1), ((1,1), 1)], # -A11 B11
    ]
    
    # Wait, the list above is messy. Let's do it programmatically.
    
    # Define inner Strassen terms for 2x2
    # Returns list of (u_row, u_col, v_row, v_col, w_row, w_col, coeff)
    # relative to the 2x2 block
    
    def inner_strassen_terms(a_ro, a_co, b_ro, b_co, coeff_outer):
        # a_ro, a_co: offset of A block in 4x4
        # b_ro, b_co: offset of B block in 4x4
        # We need to add 7 terms.
        new_terms = []
        
        # Inner M1: (X00+X11)(Y00+Y11)
        # X00 is A[a_ro, a_co]
        # X11 is A[a_ro+1, a_co+1]
        # Y00 is B[b_ro, b_co]
        # Y11 is B[b_ro+1, b_co+1]
        
        # Terms for (X00+X11)(Y00+Y11):
        # X00 Y00, X00 Y11, X11 Y00, X11 Y11
        
        # But we use Strassen for the 2x2 product too?
        # Yes, rank 7 for 2x2x2.
        # So we apply the 7 Strassen terms to the 2x2 matrices X and Y.
        
        # Let's hardcode the 7 terms for 2x2 with offsets.
        
        # 1. (X00+X11)(Y00+Y11)
        # 2. (X10+X11)Y00
        # 3. X00(Y01-Y11)
        # 4. X11(Y10-Y00)
        # 5. (X00+X01)Y11
        # 6. (X10-X00)(Y00+Y01)
        # 7. (X01-X11)(Y10+Y11)
        
        # We need to expand each of these into scalar products.
        # Example 1: (X00+X11)(Y00+Y11)
        # This is not a single scalar product. It's a product of 2x2 matrices.
        # Wait, I am confusing levels.
        
        # The "Outer Strassen" produces 7 terms, each of which is a product of two 2x2 matrices.
        # To get rank 49, we must decompose each of these 2x2 products using Strassen (rank 7).
        # So for each of the 7 outer terms, we have 7 inner terms.
        
        # Let's define a function that takes the A block offset and B block offset
        # and returns the 7 scalar terms (u, v, w, coeff) for that 2x2 product.
        
        def get_2x2_terms(a_off_r, a_off_c, b_off_r, b_off_c, scale):
            res = []
            # X is A block at (a_off_r, a_off_c)
            # Y is B block at (b_off_r, b_off_c)
            
            # 1. (X00+X11)(Y00+Y11)
            # X00: A[a_off_r, a_off_c]
            # X11: A[a_off_r+1, a_off_c+1]
            # Y00: B[b_off_r, b_off_c]
            # Y11: B[b_off_r+1, b_off_c+1]
            
            # This term is a sum of 4 scalar products if schoolbook, but we use Strassen.
            # No, Strassen replaces the 4 products with 7.
            # The 7 terms are:
            # M1 = (X00+X11)(Y00+Y11) -> This IS the first Strassen term.
            # But M1 is still a product of sums.
            # (X00+X11) is a 1x1? No, X is 2x2.
            # X00 is a scalar.
            # So (X00+X11) is a scalar sum.
            # So M1 is a scalar product of two sums.
            # It expands to 4 scalar products:
            # X00 Y00 + X00 Y11 + X11 Y00 + X11 Y11
            
            # If we use Strassen for 2x2, we don't expand M1 into 4.
            # We treat the 2x2 multiplication as a black box of rank 7.
            # The 7 terms of Strassen for 2x2 are scalar products of linear combinations of entries.
            
            # So we just need to list the 7 Strassen terms for 2x2,
            # substituting X_ij with A[a_off_r+i, a_off_c+j]
            # and Y_ij with B[b_off_r+i, b_off_c+j]
            
            # Term 1: (X00+X11)(Y00+Y11)
            # This is NOT a single scalar product.
            # Wait. Strassen's algorithm expresses the product as a linear combination of 7 products.
            # Each product is of the form (Sum of X) * (Sum of Y).
            # These are scalar products.
            
            # So for Term 1:
            # U part: X00 + X11 -> A[a_off_r, a_off_c] + A[a_off_r+1, a_off_c+1]
            # V part: Y00 + Y11 -> B[b_off_r, b_off_c] + B[b_off_r+1, b_off_c+1]
            # W part: C00? No, the result is a 2x2 matrix.
            # The term contributes to C00, C01, C10, C11.
            
            # This is getting complicated to generate dynamically without a pre-computed table.
            # I will hardcode the 49 terms.
            
            # Actually, I can just use the 2x2 Strassen decomposition I verified earlier.
            # It gives 7 terms.
            # Each term has U coeff, V coeff, and W coeffs for the 4 output elements.
            
            # Let's generate the 49 terms by iterating the 7 outer terms,
            # and for each, applying the 7 inner terms.
            
            # This is too much code to generate dynamically in the time available.
            # I will fall back to a simpler, verified rank 85 construction.
            
            # Alternative: Use the known rank 85 decomposition from literature?
            # Or just use the block decomposition with schoolbook for the 2x2 parts?
            # If I use schoolbook for 2x2x2 (rank 4), then 4x4x4 rank is 4*4=16? No.
            # Outer Strassen gives 7 terms of 2x2 products.
            # If each 2x2 product is rank 4 (schoolbook), total rank is 7*4=28.
            # This is better than 49.
            # Is 2x2x2 rank 4 valid? Yes, schoolbook.
            # So 4x4x4 rank 28.
            
            # Let's use Rank 28 for 4x4x4.
            
            pass
            
        # Let's implement the Rank 28 version for 4x4x4.
        
        pass
        
    # Given the complexity, I will provide the Rank 85 solution with Strassen(49) as it is safer.
    # I will hardcode the 49 terms.
    
    # Actually, I can just write a simple generator for the 4x4x4 Strassen terms.
    
    # Let's try the Rank 28 approach (Strassen outer, Schoolbook inner).
    # Outer Strassen: 7 terms.
    # Inner Schoolbook: 4 terms each.
    # Total 28 terms.
    
    # Outer Term 1: (A00+A11)(B00+B11)
    # Inner Schoolbook for (A00+A11) and (B00+B11):
    # X = A00+A11, Y = B00+B11
    # X00 = A000 + A110? No.
    # A00 is a 2x2 block.
    # X is a 2x2 matrix.
    # X00 = A[0,0] + A[2,2] (using 0-based 4x4 indices)
    # X01 = A[0,1] + A[2,3]
    # X10 = A[1,0] + A[2,2]? No, A11 is A[2:4, 2:4].
    # X00 = A[0,0] + A[2,2]
    # X01 = A[0,1] + A[2,3]
    # X10 = A[1,0] + A[3,2]
    # X11 = A[1,1] + A[3,3]
    
    # This is doable.
    
    # I will implement the Rank 28 for 4x4x4.
    
    return {"U": [], "V": [], "W": []}

# The above is a stub. I will write the full correct code now.

