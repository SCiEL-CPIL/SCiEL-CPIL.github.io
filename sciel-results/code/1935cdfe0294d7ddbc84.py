import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Generate Rank 7 Strassen matrices for 2x2x2
    # U, V, W are 4x7 matrices (indices 0..3 for 2x2 row-major, 0..6 for terms)
    U22 = np.zeros((4, 7), dtype=int)
    V22 = np.zeros((4, 7), dtype=int)
    W22 = np.zeros((4, 7), dtype=int)
    
    # Term 0: (A00+A11)(B00+B11) -> C00, C11
    # A: [0,0,0,1] + [0,0,1,0]? No, indices:
    # 0: A00, 1: A01, 2: A10, 3: A11
    U22[0, 0] = 1; U22[3, 0] = 1
    V22[0, 0] = 1; V22[3, 0] = 1
    W22[0, 0] = 1; W22[3, 0] = 1
    
    # Term 1: (A10+A11)B00 -> C10, C11
    U22[2, 1] = 1; U22[3, 1] = 1
    V22[0, 1] = 1
    W22[2, 1] = 1; W22[3, 1] = -1
    
    # Term 2: A00(B01-B11) -> C01, C11
    U22[0, 2] = 1
    V22[1, 2] = 1; V22[3, 2] = -1
    W22[1, 2] = 1; W22[3, 2] = 1
    
    # Term 3: A11(B10-B00) -> C00, C10
    U22[3, 3] = 1
    V22[2, 3] = 1; V22[0, 3] = -1
    W22[0, 3] = 1; W22[2, 3] = 1
    
    # Term 4: (A00+A01)B11 -> C00, C01
    U22[0, 4] = 1; U22[1, 4] = 1
    V22[3, 4] = 1
    W22[0, 4] = -1; W22[1, 4] = 1
    
    # Term 5: (A10-A00)(B00+B01) -> C11
    U22[2, 5] = 1; U22[0, 5] = -1
    V22[0, 5] = 1; V22[1, 5] = 1
    W22[3, 5] = 1
    
    # Term 6: (A01-A11)(B10+B11) -> C00
    U22[1, 6] = 1; U22[3, 6] = -1
    V22[2, 6] = 1; V22[3, 6] = 1
    W22[0, 6] = 1

    # Construct Rank 49 decomposition for 4x4x4
    # U44: 16x49, V44: 16x49, W44: 16x49
    U44 = np.zeros((16, 49), dtype=int)
    V44 = np.zeros((16, 49), dtype=int)
    W44 = np.zeros((16, 49), dtype=int)
    
    # Outer Strassen terms for 4x4
    # Each term t (0..6) combines blocks.
    # A blocks: 00, 01, 10, 11 (indices 0..15 in 4x4)
    # Block 00: rows 0,1 cols 0,1 -> idx 0,1,4,5
    # Block 01: rows 0,1 cols 2,3 -> idx 2,3,6,7
    # Block 10: rows 2,3 cols 0,1 -> idx 8,9,12,13
    # Block 11: rows 2,3 cols 2,3 -> idx 10,11,14,15
    
    # We define the outer terms by specifying which 2x2 blocks are added/subtracted.
    # This is equivalent to the 2x2 Strassen logic applied to the 2x2 grid of blocks.
    
    # Map local 2x2 indices to global 4x4 indices for a specific block
    def get_block_indices(block_idx):
        # block_idx: 0=00, 1=01, 2=10, 3=11
        r_base = (block_idx // 2) * 2
        c_base = (block_idx % 2) * 2
        indices = []
        for dr in range(2):
            for dc in range(2):
                r = r_base + dr
                c = c_base + dc
                indices.append(r * 4 + c)
        return indices

    block_indices = [get_block_indices(i) for i in range(4)]

    # Outer terms definitions (same logic as 2x2 Strassen but on blocks)
    # t_idx, a_blocks, b_blocks, c_blocks
    # a_blocks: list of (block_id, weight)
    
    outer_terms = [
        # 0: (A00+A11)(B00+B11) -> C00, C11
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # 1: (A10+A11)B00 -> C10, C11
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # 2: A00(B01-B11) -> C01, C11
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # 3: A11(B10-B00) -> C00, C10
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # 4: (A00+A01)B11 -> C00, C01
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # 5: (A10-A00)(B00+B01) -> C11
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # 6: (A01-A11)(B10+B11) -> C00
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    ]

    for t_idx in range(7):
        a_parts, b_parts, c_parts = outer_terms[t_idx]
        
        for s_idx in range(7):
            col = t_idx * 7 + s_idx
            
            # Fill U44
            # The "A" matrix for this term is sum(w_a * A_block_a).
            # The 2x2 Strassen U22[s_idx, la] gives coeff for local A index la.
            # Local A index la corresponds to a specific position in the 2x2 block.
            # We need to add w_a * U22[s_idx, la] to the global index corresponding to la in block a.
            
            for la in range(4):
                coeff_local = U22[la, s_idx]
                if coeff_local == 0: continue
                
                # Determine which global indices in the combined A matrix correspond to local index la
                # The combined A matrix is formed by placing the blocks.
                # Actually, U22 operates on a 2x2 matrix X.
                # X is a 2x2 matrix where each element is a 2x2 block? No.
                # The recursion works like this:
                # We treat the 4x4 A as a 2x2 matrix of 2x2 blocks.
                # Let A_blocks be the 2x2 grid of blocks.
                # The term s_idx in Strassen computes a product of linear combos of these blocks.
                # Wait, the standard recursive Strassen applies to the *elements*.
                # Here, we are applying Strassen to the *blocks*.
                # So, for a fixed outer term t, we form matrices A_t and B_t which are 2x2 matrices of 2x2 blocks?
                # No, A_t is a 4x4 matrix.
                # Let's trace:
                # Outer term t: A_t = sum w_a A_{block_a}.
                # B_t = sum w_b B_{block_b}.
                # We need to compute A_t B_t.
                # We use inner Strassen (s_idx) to compute A_t B_t.
                # Inner Strassen operates on 2x2 matrices.
                # A_t is 4x4. We can view A_t as a 2x2 matrix of 2x2 blocks?
                # Yes, because A_t is formed by linear combinations of the 2x2 blocks of A.
                # So A_t has the same block structure as A.
                # Let A_t blocks be A_{t,00}, A_{t,01}, A_{t,10}, A_{t,11}.
                # A_{t,ij} = sum_{a: block_a maps to ij?} No.
                # The outer Strassen combines blocks.
                # For example, Term 0: A_t = A_00 + A_11.
                # This means the top-left 2x2 block of A_t is A_00, and bottom-right is A_11. Top-right and bottom-left are 0.
                # So A_t is a 4x4 matrix.
                # Now we apply inner Strassen to A_t and B_t.
                # Inner Strassen splits A_t into four 2x2 sub-blocks.
                # Let's call them A_t^00, A_t^01, A_t^10, A_t^11.
                # A_t^00 is the top-left 2x2 of A_t.
                # Since A_t = A_00 (in TL) + A_11 (in BR),
                # A_t^00 = A_00 (which is 2x2).
                # A_t^11 = A_11 (which is 2x2).
                # A_t^01 = 0.
                # A_t^10 = 0.
                
                # So, for the inner Strassen term s_idx, we need the coefficients for the elements of A_t^00, A_t^01, etc.
                # U22[la, s_idx] is the coefficient for the element at local index la in the 2x2 matrix being multiplied.
                # The 2x2 matrix being multiplied is composed of the 2x2 blocks of A_t.
                # So, if U22[la, s_idx] != 0, it contributes to the block of A_t corresponding to the block index of la?
                # No. U22 operates on a 2x2 matrix of *scalars* (if we were doing 2x2).
                # Here, we are doing 4x4 by treating it as 2x2 of 2x2s.
                # So the "scalars" of the outer 2x2 are the 2x2 blocks.
                # The inner Strassen operates on these 2x2 blocks as if they were scalars?
                # No, the inner Strassen is a full 2x2 matrix multiplication.
                # Wait.
                # Outer step: Reduces 4x4 mult to 7 mults of 2x2 matrices?
                # No, Outer Strassen reduces 4x4 (viewed as 2x2 of 2x2s) to 7 products of 2x2 matrices.
                # Each product is of the form (Linear Combo of A_blocks) * (Linear Combo of B_blocks).
                # Let X = sum w_a A_{block_a}. Y = sum w_b B_{block_b}.
                # We need to compute X Y. X and Y are 4x4 matrices.
                # To compute X Y, we can use the Rank 7 Strassen algorithm for 2x2 matrices?
                # No, X and Y are 4x4.
                # We can view X as a 2x2 matrix of 2x2 blocks.
                # Let X_{00}, X_{01}, X_{10}, X_{11} be the 2x2 blocks of X.
                # Since X is a linear combo of the original blocks, X_{ij} is a 2x2 matrix.
                # We can apply Strassen to the 2x2 matrix of blocks?
                # Yes! Strassen works over any ring. The "elements" are 2x2 matrices.
                # So, for each outer term t, we compute 7 products of 2x2 matrices (the blocks of X and Y).
                # This requires 7 * 7 = 49 scalar multiplications?
                # No, each "product" of 2x2 matrices requires 7 scalar multiplications (using Strassen again).
                # So total 49 scalar multiplications.
                
                # So, for a fixed outer term t:
                # We have X (4x4) and Y (4x4).
                # We decompose X into 2x2 blocks: X_00, X_01, X_10, X_11.
                # We decompose Y into 2x2 blocks: Y_00, Y_01, Y_10, Y_11.
                # We apply Strassen to the 2x2 matrix of blocks.
                # The 7 terms of this inner Strassen are s_idx = 0..6.
                # For each s_idx, we compute a product P_s = (Linear Combo of X_blocks) * (Linear Combo of Y_blocks).
                # The "Linear Combo of X_blocks" is a 2x2 matrix M_X.
                # The "Linear Combo of Y_blocks" is a 2x2 matrix M_Y.
                # P_s = M_X M_Y.
                # M_X and M_Y are 2x2 matrices.
                # We need to compute M_X M_Y exactly.
                # We can use the Rank 7 Strassen decomposition for 2x2 matrices (U22, V22, W22) to compute M_X M_Y.
                # This gives 7 scalar products.
                
                # So, for a specific (t, s):
                # We need to form M_X and M_Y.
                # M_X is determined by U22[:, s].
                # M_X[la] = sum_{block_b} U22[la, s] * X_block_b ?
                # No. U22[la, s] is the coefficient for the element at position la in the 2x2 matrix M_X.
                # The elements of M_X are the 2x2 blocks of X.
                # So M_X[la] is a 2x2 matrix which is the block of X at position la.
                # X_block_la is the 2x2 block of X at position la.
                # X_block_la = sum_{a} w_{t,a,la} A_{block_a} ?
                # Recall X = sum_a w_{t,a} A_{block_a}.
                # The block of X at position la (0..3) is sum_a w_{t,a} * (block of A_{block_a} at position la).
                # But A_{block_a} is a 2x2 matrix placed at a specific location in the 4x4 A.
                # It is zero elsewhere.
                # So, the block of X at position la is non-zero only if A_{block_a} occupies that block position.
                # Actually, A_{block_a} IS the block at position a.
                # So, if we sum w_a A_{block_a}, the resulting matrix X has:
                # X_block_00 = w_0 A_00 (if a=0 contributes to 00)
                # Wait, A_{block_0} is the top-left 2x2. So it contributes to X_block_00.
                # A_{block_1} is top-right. Contributes to X_block_01.
                # So, X_block_la = w_{t, la} A_{block_la} ?
                # Only if the outer term t uses the block at position la.
                # In Term 0: X = A_00 + A_11.
                # X_block_00 = A_00.
                # X_block_11 = A_11.
                # X_block_01 = 0.
                # X_block_10 = 0.
                
                # So, for a given outer term t, we know which blocks are present and their weights.
                # Let's define `x_block_weights[la]` = weight of A_block_la in X.
                # From `outer_terms[t][0]` (a_parts), we have list of (block_id, weight).
                # So `x_block_weights[block_id] += weight`.
                
                # Similarly for Y.
                
                # Now, for inner term s:
                # M_X is a 2x2 matrix where M_X[la] = x_block_weights[la] * A_block_la ?
                # No. M_X[la] is the element of the 2x2 matrix M_X at index la.
                # The element is a 2x2 matrix.
                # This 2x2 matrix is `x_block_weights[la] * A_block_la`.
                # Note: `A_block_la` is the original 2x2 block of A.
                # So, M_X[la] is a 2x2 matrix.
                # We need to compute the product of M_X and M_Y (2x2 matrices of 2x2 matrices).
                # We use U22, V22, W22 to decompose this 2x2 matrix multiplication.
                # The "scalar" multiplication is actually 2x2 matrix multiplication?
                # No, U22, V22, W22 decompose the multiplication of 2x2 matrices of *scalars*.
                # Here, our "scalars" are 2x2 matrices.
                # So we need to apply the Strassen structure again?
                # Yes. The "inner" Strassen is for 2x2 matrices of "scalars".
                # But our "scalars" are 2x2 matrices.
                # So we need to decompose the multiplication of 2x2 matrices of 2x2 matrices.
                # This is just applying Strassen to the 2x2 grid, where each "element" is a 2x2 matrix.
                # To multiply two 2x2 matrices of 2x2 matrices, we can use the Rank 7 Strassen algorithm, where each "scalar product" is a 2x2 matrix product.
                # But we want a full tensor decomposition into scalar products.
                # So we need to expand each "2x2 matrix product" into 7 scalar products using Strassen again.
                
                # So, for a fixed (t, s):
                # We have M_X (2x2 of 2x2s) and M_Y (2x2 of 2x2s).
                # We apply Strassen to M_X, M_Y.
                # The 7 terms of this Strassen are u_idx = 0..6.
                # For each u_idx:
                #   We compute a product of two linear combinations of the "elements" of M_X and M_Y.
                #   Let L_X = sum_{la} U22[la, u_idx] M_X[la].
                #   Let L_Y = sum_{lb} V22[lb, u_idx] M_Y[lb].
                #   L_X and L_Y are 2x2 matrices.
                #   We need to compute L_X L_Y.
                #   We use Strassen again (v_idx = 0..6) to decompose L_X L_Y.
                #   For each v_idx:
                #      We compute scalar products.
                
                # This leads to 7 (outer) * 7 (middle) * 7 (inner) = 343 terms?
                # No.
                # Let's re-evaluate the hierarchy.
                # 4x4 multiplication.
                # Level 1: Strassen on 2x2 blocks. 7 terms.
                # Each term involves multiplying two 4x4 matrices? No, two linear combos of blocks.
                # The result of each term is a 4x4 matrix.
                # To compute the product of two 4x4 matrices (which are sparse/structured), we can just do it directly?
                # No, we want a tensor decomposition.
                
                # Let's stick to the simplest correct implementation:
                # Rank 49 for 4x4x4 is standard.
                # It is constructed by 7 outer terms, each requiring 7 inner terms.
                # The inner terms are scalar multiplications.
                # How?
                # The outer term t produces matrices X_t and Y_t (4x4).
                # We need to compute X_t Y_t.
                # X_t and Y_t are 4x4.
                # We can use the Rank 7 Strassen decomposition for 2x2 matrices, but applied to the 2x2 blocks of X_t and Y_t.
                # This requires 7 multiplications of 2x2 matrices.
                # Each 2x2 matrix multiplication can be done with 7 scalar multiplications (Strassen).
                # So 7 * 7 = 49 scalar multiplications.
                
                # So, for each outer term t:
                #   We form X_t and Y_t.
                #   We apply Strassen to the 2x2 grid of blocks of X_t and Y_t.
                #   This gives 7 intermediate products P_s (s=0..6), where P_s is a 2x2 matrix.
                #   Each P_s is formed by multiplying two 2x2 matrices M_X_s and M_Y_s.
                #   M_X_s and M_Y_s are 2x2 matrices formed by linear combos of the 2x2 blocks of X_t and Y_t.
                #   To compute M_X_s M_Y_s, we use Strassen again (7 scalar products).
                
                # So, total terms = 7 (outer) * 7 (middle) * 7 (inner) = 343?
                # No, the "middle" Strassen produces 2x2 matrices.
                # The "inner" Strassen decomposes the multiplication of those 2x2 matrices.
                # So yes, 7*7*7 = 343 terms for 4x4?
                # But Rank 49 is known for 4x4.
                # Why?
                # Because the "middle" step doesn't require 7 full 2x2 multiplications.
                # The outer Strassen gives 7 terms.
                # Each term is a product of two 4x4 matrices?
                # No, each term is a product of two linear combinations of blocks.
                # Let X_t = sum w_a A_{block_a}.
                # X_t is a 4x4 matrix.
                # We need to compute X_t Y_t.
                # If we use naive multiplication, it's 16 products.
                # If we use Strassen on the 2x2 blocks of X_t and Y_t, it's 7 products of 2x2 matrices.
                # Each 2x2 matrix product is 7 scalar products.
                # Total 49.
                
                # So, for each outer term t:
                #   We have X_t and Y_t.
                #   We decompose X_t into blocks X_t_00, X_t_01, X_t_10, X_t_11.
                #   We decompose Y_t into blocks Y_t_00, Y_t_01, Y_t_10, Y_t_11.
                #   We apply Strassen to these blocks.
                #   The 7 terms of this Strassen are s_idx = 0..6.
                #   For each s_idx, we compute a product of two 2x2 matrices.
                #   Let A_s = Linear Combo of X_t blocks.
                #   Let B_s = Linear Combo of Y_t blocks.
                #   A_s and B_s are 2x2 matrices.
                #   We need to compute A_s B_s.
                #   We use Strassen (U22, V22, W22) to decompose A_s B_s into 7 scalar products.
                #   So for each (t, s_idx), we have 7 scalar products.
                #   Total 7 * 7 = 49.

    # Implementation Plan:
    # 1. Define outer terms (t=0..6) with block weights.
    # 2. For each t, determine X_t blocks and Y_t blocks.
    #    X_t_block[la] = weight * A_block[la] (if la is in the sum, else 0).
    #    Actually, X_t is the sum. So X_t_block[la] is the block of X_t at position la.
    #    X_t_block[la] = w_{t,la} * A_block[la].
    # 3. Apply middle Strassen (U22, V22, W22) to the 2x2 matrices of blocks.
    #    For each middle term s (0..6):
    #       A_s = sum_{la} U22[la, s] X_t_block[la].
    #       B_s = sum_{lb} V22[lb, s] Y_t_block[lb].
    #       A_s and B_s are 2x2 matrices.
    #       Note: A_s is a linear combination of 2x2 matrices.
    #       A_s = sum_{la} U22[la, s] * (w_{t,la} * A_block[la]).
    #       A_s = sum_{la} (U22[la, s] * w_{t,la}) * A_block[la].
    #       So A_s is a 2x2 matrix. Its elements are linear combinations of elements of A_block[la].
    #       Specifically, A_s[i,j] = sum_{la} (U22[la, s] * w_{t,la}) * A_block[la][i,j].
    # 4. Apply inner Strassen to A_s and B_s.
    #    For each inner term u (0..6):
    #       We need scalar products.
    #       The inner Strassen U22, V22, W22 operates on 2x2 matrices.
    #       The "scalar" at index la in A_s is A_s[la].
    #       A_s[la] is a scalar (element of the 2x2 matrix A_s).
    #       Wait, A_s is a 2x2 matrix.
    #       The inner Strassen decomposes the multiplication of two 2x2 matrices.
    #       The terms are u=0..6.
    #       For each u, we have a product of two linear combos of the elements of A_s and B_s.
    #       Let L_A = sum_{la} U22[la, u] A_s[la].
    #       Let L_B = sum_{lb} V22[lb, u] B_s[lb].
    #       L_A and L_B are scalars.
    #       The product is L_A L_B.
    #       We add this to the U, V, W matrices.
        
    # So, for a fixed (t, s, u):
    #   L_A = sum_{la} U22[la, u] A_s[la]
    #   A_s[la] = sum_{block_a} (U22[la, s] * w_{t,block_a}) * A_block[block_a][la]
    #   So L_A = sum_{la} U22[la, u] sum_{block_a} (U22[la, s] * w_{t,block_a}) A_block[block_a][la]
    #   L_A = sum_{block_a} w_{t,block_a} sum_{la} U22[la, u] U22[la, s] A_block[block_a][la]
    #   This is a linear combination of elements of A.
    #   Similarly for L_B.
    #   The product L_A L_B is a scalar.
    #   We need to express L_A and L_B as linear combinations of global A and B indices.
    
    # Let's implement this directly.

    U44 = np.zeros((16, 49), dtype=int)
    V44 = np.zeros((16, 49), dtype=int)
    W44 = np.zeros((16, 49), dtype=int)

    for t in range(7):
        a_parts, b_parts, c_parts = outer_terms[t]
        
        # Determine weights for blocks in X_t and Y_t
        x_weights = np.zeros(4, dtype=int)
        for b_idx, w in a_parts:
            x_weights[b_idx] += w
            
        y_weights = np.zeros(4, dtype=int)
        for b_idx, w in b_parts:
            y_weights[b_idx] += w
            
        for s in range(7):
            # A_s and B_s are 2x2 matrices
            # A_s[la] = sum_{block_a} x_weights[block_a] * U22[la, s] * A_block[block_a][la]
            # B_s[lb] = sum_{block_b} y_weights[block_b] * V22[lb, s] * B_block[block_b][lb]
            
            for u in range(7):
                col = t * 49 + s * 7 + u
                
                # Compute L_A coefficients for global A indices
                # L_A = sum_{la} U22[la, u] A_s[la]
                # L_A = sum_{la} U22[la, u] sum_{block_a} x_weights[block_a] U22[la, s] A_block[block_a][la]
                # L_A = sum_{block_a} x_weights[block_a] sum_{la} U22[la, u] U22[la, s] A_block[block_a][la]
                
                for block_a in range(4):
                    w_a = x_weights[block_a]
                    if w_a == 0: continue
                    
                    for la in range(4):
                        coeff_la = U22[la, u] * U22[la, s]
                        if coeff_la == 0: continue
                        
                        # Add to global A indices in block_a
                        for g_idx in block_indices[block_a]:
                            U44[g_idx, col] += w_a * coeff_la

                # Compute L_B coefficients for global B indices
                for block_b in range(4):
                    w_b = y_weights[block_b]
                    if w_b == 0: continue
                    
                    for lb in range(4):
                        coeff_lb = V22[lb, u] * V22[lb, s]
                        if coeff_lb == 0: continue
                        
                        for g_idx in block_indices[block_b]:
                            V44[g_idx, col] += w_b * coeff_lb

                # Compute W coefficients
                # W22[lc, u] is coeff for local C index lc in the 2x2 result of A_s B_s.
                # The result of the inner Strassen term u is added to the 2x2 matrix P_s.
                # P_s[la] gets contribution W22[la, u] * L_A * L_B.
                # P_s is a 2x2 matrix.
                # P_s contributes to the final C.
                # The outer Strassen term t adds P_s to C blocks according to c_parts.
                # c_parts is list of (block_c, weight).
                # So C_block_c gets weight * P_s.
                # P_s[la] is a scalar.
                # It contributes to C_block_c[la].
                
                for lc in range(4):
                    coeff_lc = W22[lc, u]
                    if coeff_lc == 0: continue
                    
                    for c_block, w_c in c_parts:
                        # Add to global C indices in block_c at local index lc
                        # Local index lc in 2x2 block.
                        # Global index in C_block_c.
                        for g_idx in block_indices[c_block]:
                            # Check if g_idx corresponds to local index lc in c_block
                            # block_indices[c_block] returns global indices in row-major order.
                            # lc is also row-major index in 2x2.
                            # So the k-th element in block_indices[c_block] corresponds to local index k.
                            # We need to match lc.
                            # Since block_indices returns 4 indices, and lc is 0..3,
                            # we can just use the index in the list.
                            # But we need to be sure the order matches.
                            # get_block_indices returns indices in row-major order of the 2x2 block.
                            # So block_indices[c_block][lc] is the global index.
                            g_idx_lc = block_indices[c_block][lc]
                            W44[g_idx_lc, col] += w_c * coeff_lc

    # Now we have U44, V44, W44 for 4x4x4.
    # Construct 4x4x8 by concatenating two 4x4x4 decompositions.
    
    rank = 98
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((32, rank), dtype=int)
    W = np.zeros((32, rank), dtype=int)
    
    # First half (cols 0-3 of B)
    U[:, 0:49] = U44
    # V mapping: 4x4 B -> 4x8 B.
    # V44 indices 0..15 correspond to j*4+k.
    # Global V indices j*8+k.
    for v_idx_44 in range(16):
        j = v_idx_44 // 4
        k = v_idx_44 % 4
        v_idx_global = j * 8 + k
        V[v_idx_global, 0:49] = V44[v_idx_44]
        
    # W mapping: 4x4 C -> 4x8 C.
    # W44 indices 0..15 correspond to k*4+i.
    # Global W indices k*4+i.
    # For first half, k=0..3. Indices are same.
    W[0:16, 0:49] = W44
    
    # Second half (cols 4-7 of B)
    U[:, 49:98] = U44
    
    for v_idx_44 in range(16):
        j = v_idx_44 // 4
        k = v_idx_44 % 4
        k_global = k + 4
        v_idx_global = j * 8 + k_global
        V[v_idx_global, 49:98] = V44[v_idx_44]
        
    for w_idx_44 in range(16):
        k = w_idx_44 // 4
        i = w_idx_44 % 4
        k_global = k + 4
        w_idx_global = k_global * 4 + i
        W[w_idx_global, 49:98] = W44[w_idx_44]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
