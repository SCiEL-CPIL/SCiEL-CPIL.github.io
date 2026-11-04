import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    # Rank 49 for 4x4x4 + 16 for 4x4x1 = 65
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
        
    # Part 1: 4x4x4 multiplication using Recursive Strassen (Rank 49)
    # We need to define the Strassen decomposition for 2x2 and apply it recursively.
    
    # Base Strassen 2x2 terms
    # Returns list of (U_coeffs, V_coeffs, W_coeffs) for 2x2x2
    # Indices for 2x2:
    # A: 00, 01, 10, 11 -> indices 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> indices 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> indices 0, 1, 2, 3
    
    def get_strassen_2x2():
        terms = []
        # M1 = (A00+A11)(B00+B11)
        u = [1, 0, 0, 1]
        v = [1, 0, 0, 1]
        w = [1, 0, 0, 1] # C00 += M1, C11 += M1
        terms.append((u, v, w))
        
        # M2 = (A10+A11)B00
        u = [0, 0, 1, 1]
        v = [1, 0, 0, 0]
        w = [0, 0, 1, -1] # C10 += M2, C11 -= M2
        terms.append((u, v, w))
        
        # M3 = A00(B01-B11)
        u = [1, 0, 0, 0]
        v = [0, 1, 0, -1]
        w = [0, 1, 0, 1] # C01 += M3, C11 += M3
        terms.append((u, v, w))
        
        # M4 = A11(B10-B00)
        u = [0, 0, 0, 1]
        v = [0, 0, 1, -1]
        w = [1, 0, 1, 0] # C00 += M4, C10 += M4
        terms.append((u, v, w))
        
        # M5 = (A00+A01)B11
        u = [1, 1, 0, 0]
        v = [0, 0, 0, 1]
        w = [1, 1, 0, 0] # C00 -= M5, C01 += M5
        terms.append((u, v, w))
        
        # M6 = (A10-A00)(B00+B01)
        u = [0, 0, 1, -1] # Wait, A10 is idx 2, A00 is idx 0.
        # Correction: u = [ -1, 0, 1, 0 ]
        u = [-1, 0, 1, 0]
        v = [1, 1, 0, 0]
        w = [0, 0, 0, 1] # C11 += M6
        terms.append((u, v, w))
        
        # M7 = (A01-A11)(B10+B11)
        u = [0, 1, 0, -1] # A01 is idx 1, A11 is idx 3
        v = [0, 0, 1, 1]
        w = [1, 0, 0, 0] # C00 += M7
        terms.append((u, v, w))
        
        return terms

    strassen_2x2_terms = get_strassen_2x2()
    
    # Recursive Strassen for 4x4
    # A is 4x4. Split into 2x2 blocks.
    # Block indices: 00 (0,0), 01 (0,1), 10 (1,0), 11 (1,1)
    # Each block is 2x2.
    
    # Outer Strassen combinations for blocks:
    # X1 = A00 + A11
    # X2 = A10 + A11
    # X3 = A00
    # X4 = A11
    # X5 = A00 + A01
    # X6 = A10 - A00
    # X7 = A01 - A11
    
    # Y1 = B00 + B11
    # Y2 = B00
    # Y3 = B01 - B11
    # Y4 = B10 - B00
    # Y5 = B11
    # Y6 = B00 + B01
    # Y7 = B10 + B11
    
    # For each t in 1..7:
    #   Compute P_t = X_t Y_t (2x2 matrix product)
    #   Decompose P_t using strassen_2x2_terms
    #   Each term in strassen_2x2_terms gives a scalar product of linear combos of elements of X_t and Y_t.
    #   Map these back to A and B elements.
    #   Determine contribution to C blocks.
    
    # Mapping of block indices to element indices in 4x4 matrix (row-major)
    # Block 00: rows 0,1 cols 0,1 -> elements 0,1,4,5
    # Block 01: rows 0,1 cols 2,3 -> elements 2,3,6,7
    # Block 10: rows 2,3 cols 0,1 -> elements 8,9,12,13
    # Block 11: rows 2,3 cols 2,3 -> elements 10,11,14,15
    
    # Let's define the linear combinations of A blocks for X_t
    # X_t is a 2x2 matrix of coefficients for the blocks? No, X_t is a 2x2 matrix of A-blocks.
    # So X_t has elements (X_t)_{ij} which are linear combos of A_{block}.
    
    # We represent a 2x2 block matrix by a 4x4 matrix of coefficients relative to A's elements?
    # No, it's easier to just generate the U vector directly.
    
    # For a given outer term t:
    # X_t is defined by coefficients on A00, A01, A10, A11.
    # Let coeff_X[t][block_idx] be the coefficient.
    # Similarly for Y_t.
    
    # But X_t is a 2x2 matrix.
    # X1 = [[A00, 0], [0, A11]]? No.
    # Strassen on blocks:
    # A = [[A00, A01], [A10, A11]]
    # X1 = A00 + A11 (This is a 2x2 matrix sum)
    # So X1 is a 2x2 matrix where each element is a sum of elements from A00 and A11.
    
    # Let's define the structure of X_t and Y_t as 2x2 matrices of "block-linear-combinations".
    # Actually, we can just track the coefficients for each of the 16 elements of A.
    
    # For each outer term t (1..7):
    #   We have 7 inner terms.
    #   For each inner term s (1..7):
    #      The inner term s computes a scalar product $u_s \cdot v_s$ where $u_s$ is a linear combo of elements of $X_t$ and $v_s$ is a linear combo of elements of $Y_t$.
    #      $X_t$ is a linear combo of A's blocks.
    #      So $u_s$ is a linear combo of A's elements.
    #      Similarly for $v_s$ and B's elements.
    #      The result contributes to a specific element of $P_t$ (a 2x2 matrix).
    #      $P_t$ contributes to C's blocks via outer Strassen reconstruction.
    
    # Let's implement this mapping.
    
    # Outer Strassen Coefficients for X_t (on A blocks)
    # t=1: X1 = A00 + A11. So X1[0,0] = A00, X1[1,1] = A11? No, X1 is a 2x2 matrix.
    # In Strassen, $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$.
    # Here $A_{00}$ is a block. So $X_1 = A_{00} + A_{11}$. This is a 2x2 matrix.
    # The elements of $X_1$ are $(A_{00}+A_{11})_{ij}$.
    # So $X_1$ has the same shape as A's blocks.
    
    # Let's define the coefficient vectors for X_t relative to A's 16 elements.
    # A's elements are indexed 0..15.
    # Block 00: 0,1,4,5
    # Block 01: 2,3,6,7
    # Block 10: 8,9,12,13
    # Block 11: 10,11,14,15
    
    # For X1 = A00 + A11:
    # Coeff for A00 elements is 1, for A11 elements is 1, others 0.
    # But X1 is a 2x2 matrix.
    # When we apply inner Strassen to X1 * Y1, we treat X1 as a 2x2 matrix.
    # The inner Strassen term s uses specific elements of X1.
    # E.g., inner M1 uses (X1)00 + (X1)11.
    # (X1)00 is the top-left element of the block sum A00+A11.
    # (X1)00 = (A00)00 + (A11)00 = A[0,0] + A[2,2].
    
    # This is getting complex to hardcode.
    # Alternative: Just generate the 49 terms by iterating through the recursive structure.
    
    # Let's write a helper to get the U vector for a specific path.
    
    # We'll represent the linear combination of A elements as a list of 16 floats.
    
    def get_outer_X_coeffs(t):
        # Returns a 2x2 matrix of coefficients for A blocks?
        # No, returns a 4x4 matrix of coefficients for A elements?
        # X_t is a 2x2 matrix of A-blocks.
        # So X_t[i,j] is a linear combo of A-blocks.
        # But A-blocks are 2x2 matrices.
        # So X_t is effectively a 4x4 matrix of coefficients for A elements.
        
        # Let's just build the 4x4 coefficient matrix for X_t.
        coeff = [[0.0]*4 for _ in range(4)]
        
        if t == 1: # A00 + A11
            # A00 is top-left 2x2, A11 is bottom-right 2x2
            for i in range(2):
                for j in range(2):
                    coeff[i][j] = 1.0 # From A00
                    coeff[i+2][j+2] = 1.0 # From A11
        elif t == 2: # A10 + A11
            for i in range(2):
                for j in range(2):
                    coeff[i+2][j] = 1.0 # From A10
                    coeff[i+2][j+2] = 1.0 # From A11
        elif t == 3: # A00
            for i in range(2):
                for j in range(2):
                    coeff[i][j] = 1.0
        elif t == 4: # A11
            for i in range(2):
                for j in range(2):
                    coeff[i+2][j+2] = 1.0
        elif t == 5: # A00 + A01
            for i in range(2):
                for j in range(2):
                    coeff[i][j] = 1.0 # A00
                    coeff[i][j+2] = 1.0 # A01
        elif t == 6: # A10 - A00
            for i in range(2):
                for j in range(2):
                    coeff[i][j] = -1.0 # -A00
                    coeff[i+2][j] = 1.0 # A10
        elif t == 7: # A01 - A11
            for i in range(2):
                for j in range(2):
                    coeff[i][j+2] = 1.0 # A01
                    coeff[i+2][j+2] = -1.0 # -A11
        return coeff

    def get_outer_Y_coeffs(t):
        coeff = [[0.0]*4 for _ in range(4)]
        if t == 1: # B00 + B11
            for i in range(2):
                for j in range(2):
                    coeff[i][j] = 1.0
                    coeff[i+2][j+2] = 1.0
        elif t == 2: # B00
            for i in range(2):
                for j in range(2):
                    coeff[i][j] = 1.0
        elif t == 3: # B01 - B11
            for i in range(2):
                for j in range(2):
                    coeff[i][j+2] = 1.0
                    coeff[i+2][j+2] = -1.0
        elif t == 4: # B10 - B00
            for i in range(2):
                for j in range(2):
                    coeff[i+2][j] = 1.0
                    coeff[i][j] = -1.0
        elif t == 5: # B11
            for i in range(2):
                for j in range(2):
                    coeff[i+2][j+2] = 1.0
        elif t == 6: # B00 + B01
            for i in range(2):
                for j in range(2):
                    coeff[i][j] = 1.0
                    coeff[i][j+2] = 1.0
        elif t == 7: # B10 + B11
            for i in range(2):
                for j in range(2):
                    coeff[i+2][j] = 1.0
                    coeff[i+2][j+2] = 1.0
        return coeff

    # Outer Strassen Reconstruction Coefficients for C blocks
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # P_t is a 2x2 matrix.
    # C_block is a 2x2 matrix.
    
    # For a specific inner term s in P_t:
    # It contributes to a specific element (r,c) of P_t.
    # The coefficient is given by the inner Strassen W vector.
    # This element (r,c) of P_t contributes to C_block[r,c] (which is a 2x2 block of C).
    # The coefficient is given by the outer Strassen reconstruction.
    
    # Inner Strassen W vectors (for 2x2 result)
    # idx 0: C00, 1: C01, 2: C10, 3: C11
    inner_w = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Outer Reconstruction Coefficients
    # outer_w[block_r][block_c][t]
    outer_w = [
        [ # Block Row 0
            [1, 0, 0, 1, -1, 0, 1], # C00: P1, P4, -P5, P7
            [0, 0, 1, 0, 1, 0, 0]   # C01: P3, P5
        ],
        [ # Block Row 1
            [0, 1, 0, 1, 0, 0, 0],  # C10: P2, P4
            [1, -1, 1, 0, 0, 1, 0]  # C11: P1, -P2, P3, P6
        ]
    ]
    
    # Generate 49 terms
    for t in range(7):
        X_coeff = get_outer_X_coeffs(t+1)
        Y_coeff = get_outer_Y_coeffs(t+1)
        
        for s in range(7):
            # Inner Strassen term s
            # Inner U and V vectors (length 4)
            inner_u = strassen_2x2_terms[s][0]
            inner_v = strassen_2x2_terms[s][1]
            
            # Construct full U vector (length 16)
            # U_full[i,j] = sum_{r,c} inner_u[r,c] * X_coeff[r,c] ?
            # No. X_t is a 4x4 matrix of coefficients.
            # X_t[r,c] is the coefficient of A[r,c] in the (r,c)-th element of the block-matrix X_t.
            # Wait, X_t is a 2x2 matrix of blocks.
            # X_t[block_r, block_c] is a 2x2 matrix of coefficients for A elements.
            # My get_outer_X_coeffs returns a 4x4 matrix where coeff[i,j] is the coefficient of A[i,j] in X_t.
            # So X_t[i,j] is the coefficient.
            
            # The inner Strassen term s uses a linear combination of elements of X_t.
            # inner_u is a vector of 4 coeffs for the 4 elements of the 2x2 matrix X_t.
            # But X_t is 4x4? No, X_t is a 2x2 block matrix, so it's 4x4 elements.
            # The inner Strassen treats X_t as a 2x2 matrix of "blocks"?
            # No, the inner Strassen is for 2x2 matrices.
            # So X_t must be viewed as a 2x2 matrix.
            # But X_t is formed from 2x2 blocks.
            # So the "elements" of the 2x2 matrix X_t are the 2x2 blocks.
            # This doesn't work directly with the 4x4 coefficient matrix.
            
            # Correction:
            # The recursive step multiplies 2x2 block matrices.
            # The "multiplication" is of 2x2 matrices.
            # So the inner Strassen operates on 2x2 matrices.
            # The elements of these 2x2 matrices are the 2x2 blocks of A.
            # So inner_u[0] applies to Block 00 of X_t.
            # Block 00 of X_t is a 2x2 matrix of coefficients for A's Block 00 elements.
            
            # So, U_full is constructed by:
            # For each block position (br, bc) in the 2x2 block grid:
            #   coeff_block = inner_u[br*2+bc]
            #   For each element (i,j) in that 2x2 block:
            #       U_full[br*2+i, bc*2+j] += coeff_block * X_coeff[br*2+i, bc*2+j]
            
            u_full = [0.0] * 16
            v_full = [0.0] * 20 # B is 4x5, but we only use first 4 cols here? No, V is 4x5.
            # For the 4x4x4 part, we only use B's first 4 columns.
            # V indices: j*5 + k. j in 0..3, k in 0..3.
            
            for br in range(2):
                for bc in range(2):
                    block_idx = br * 2 + bc
                    coeff_b = inner_u[block_idx]
                    if coeff_b != 0:
                        for i in range(2):
                            for j in range(2):
                                r = br * 2 + i
                                c = bc * 2 + j
                                u_full[r * 4 + c] += coeff_b * X_coeff[r][c]
                    
                    coeff_v = inner_v[block_idx]
                    if coeff_v != 0:
                        for i in range(2):
                            for j in range(2):
                                r = br * 2 + i
                                c = bc * 2 + j
                                # V index: row r, col c (for B's 4x4 part)
                                v_full[r * 5 + c] += coeff_v * Y_coeff[r][c]
            
            # Construct W vector (length 20)
            # W indices: k*4 + i. k in 0..3 (for this part), i in 0..3.
            w_full = [0.0] * 20
            
            # Inner W vector tells contribution to P_t's 2x2 elements.
            # inner_w[s] is length 4.
            # P_t[block_r, block_c] gets contribution inner_w[s][block_r*2+block_c].
            # This P_t element contributes to C's blocks via outer_w.
            
            for br in range(2):
                for bc in range(2):
                    block_idx = br * 2 + bc
                    contrib = inner_w[s][block_idx]
                    if contrib != 0:
                        # Add to C's block (br, bc)
                        for i in range(2):
                            for j in range(2):
                                r = br * 2 + i
                                c = bc * 2 + j
                                # C index: c*4 + r ? No, W index is k*n + i.
                                # k is col of C (0..3), i is row of C (0..3).
                                # So index is c * 4 + r.
                                w_full[c * 4 + r] += contrib * outer_w[br][bc][t]
            
            add_term(u_full, v_full, w_full)
            
    # Part 2: 4x4x1 (Last column)
    # Schoolbook: 16 terms
    for i in range(4):
        for j in range(4):
            u = [0.0] * 16
            v = [0.0] * 20
            w = [0.0] * 20
            
            u[i * 4 + j] = 1.0
            v[j * 5 + 4] = 1.0 # k=4
            w[4 * 4 + i] = 1.0 # k=4, i=i
            
            add_term(u, v, w)
            
    # Convert to lists of lists
    U = [row[:] for row in zip(*U_list)] if U_list else []
    # Wait, U_list contains vectors of length 16.
    # We need U to be shape (16, Rank).
    # So U[i][c] = U_list[c][i].
    
    rank = len(U_list)
    U_mat = [[0.0] * rank for _ in range(16)]
    V_mat = [[0.0] * rank for _ in range(20)]
    W_mat = [[0.0] * rank for _ in range(20)]
    
    for c in range(rank):
        for i in range(16):
            U_mat[i][c] = U_list[c][i]
        for j in range(20):
            V_mat[j][c] = V_list[c][j]
        for k in range(20):
            W_mat[k][c] = W_list[c][k]
            
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
