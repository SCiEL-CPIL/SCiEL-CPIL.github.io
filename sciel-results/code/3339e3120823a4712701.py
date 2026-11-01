"""
Strassen-based rank minimization for 4x4 matrix multiplication.
Uses recursive Strassen: 7 products of 2x2 blocks, each 2x2 product uses rank-7 Strassen.
Total rank: 7 * 7 = 49.
"""

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=4
    # Rank is 49
    rank = 49
    
    # U has shape (n*m, rank) = (16, 49)
    # V has shape (m*p, rank) = (16, 49)
    # W has shape (p*n, rank) = (16, 49)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 coefficients for U, V, W
    # Indices: 0..6
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A01-A11)*(B10+B11)
    # M7 = (A10-A00)*(B00+B01)
    
    # Coefficients for A part (U)
    # A is 2x2 block. A_{ij} corresponds to row i*2 + j? No, A is 4x4.
    # In the recursive step, we treat A as 2x2 blocks of 2x2.
    # Let's define the local 2x2 indices for the top-level Strassen.
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2.
    # Global indices for A:
    # A00: rows 0,1 cols 0,1
    # A01: rows 0,1 cols 2,3
    # A10: rows 2,3 cols 0,1
    # A11: rows 2,3 cols 2,3
    
    # We need to map the Strassen terms to the global U, V, W matrices.
    # The decomposition is:
    # C = sum_{s=0}^{6} sum_{t=0}^{6} U_{s,t} * V_{s,t} * W_{s,t}
    # where U, V, W are linear combinations of A, B, C elements.
    
    # Let's define the linear forms for the top-level Strassen (4x4 -> 2x2 blocks)
    # and the bottom-level Strassen (2x2 -> scalars).
    
    # Top-level Strassen terms (indices 0..6)
    # A-blocks: A00, A01, A10, A11
    # B-blocks: B00, B01, B10, B11
    
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A01-A11)*(B10+B11)
    # M7 = (A10-A00)*(B00+B01)
    
    # For each top-level term s (0..6), we have a product of two 2x2 matrices.
    # Let's say the term is X_s * Y_s.
    # X_s is a 2x2 matrix formed from A-blocks.
    # Y_s is a 2x2 matrix formed from B-blocks.
    # The product X_s * Y_s is a 2x2 matrix of 2x2 blocks.
    # Wait, the product is computed exactly.
    # The contribution to C is X_s * Y_s.
    # Since X_s and Y_s are 2x2 matrices of 2x2 blocks, their product is a 2x2 matrix of 2x2 blocks.
    # We need to decompose this 2x2 block multiplication into rank-7 scalar multiplications.
    
    # Actually, it's simpler to view the whole thing as a tensor.
    # The 4x4 multiplication tensor is the Kronecker product of the 2x2 multiplication tensor with itself?
    # No, Strassen is not a Kronecker product. It's a specific decomposition.
    
    # Let's construct the matrices explicitly.
    # We iterate over the 7 top-level Strassen products.
    # For each, we determine the linear combination of A elements (U) and B elements (V).
    # Then we multiply these two 2x2 matrices (symbolically) to get the contribution to C.
    # Then we decompose that 2x2 matrix multiplication using the 7 bottom-level Strassen terms.
    
    # Helper to get global index for A[i,j]
    def a_idx(i, j):
        return i * 4 + j
        
    # Helper to get global index for B[i,j]
    def b_idx(i, j):
        return i * 4 + j
        
    # Helper to get global index for C[i,j]
    def c_idx(i, j):
        return i * 4 + j

    # Define the top-level Strassen terms for A and B (2x2 blocks)
    # A_blocks:
    # A00: [[A00, A01], [A02, A03]] -> No, A is 4x4.
    # A = [[A00, A01, A02, A03],
    #      [A10, A11, A12, A13],
    #      [A20, A21, A22, A23],
    #      [A30, A31, A32, A33]]
    # Blocks:
    # A00 = [[A00, A01], [A10, A11]]
    # A01 = [[A02, A03], [A12, A13]]
    # A10 = [[A20, A21], [A30, A31]]
    # A11 = [[A22, A23], [A32, A33]]
    
    # We represent the linear forms as dictionaries: { (i,j): coeff }
    
    def add_dict(d1, d2):
        res = {}
        for k, v in d1.items():
            res[k] = res.get(k, 0) + v
        for k, v in d2.items():
            res[k] = res.get(k, 0) + v
        # Remove zeros
        return {k: v for k, v in res.items() if v != 0}

    def sub_dict(d1, d2):
        res = {}
        for k, v in d1.items():
            res[k] = res.get(k, 0) + v
        for k, v in d2.items():
            res[k] = res.get(k, 0) - v
        return {k: v for k, v in res.items() if v != 0}

    # Define the 4 A-blocks as dicts of scalar indices
    A00 = {(0,0):1, (0,1):1, (1,0):1, (1,1):1} # Wait, indices are (row, col) in 4x4
    # A00 block covers rows 0-1, cols 0-1
    A00 = {(0,0):1, (0,1):1, (1,0):1, (1,1):1}
    A01 = {(0,2):1, (0,3):1, (1,2):1, (1,3):1}
    A10 = {(2,0):1, (2,1):1, (3,0):1, (3,1):1}
    A11 = {(2,2):1, (2,3):1, (3,2):1, (3,3):1}
    
    B00 = {(0,0):1, (0,1):1, (1,0):1, (1,1):1}
    B01 = {(0,2):1, (0,3):1, (1,2):1, (1,3):1}
    B10 = {(2,0):1, (2,1):1, (3,0):1, (3,1):1}
    B11 = {(2,2):1, (2,3):1, (3,2):1, (3,3):1}
    
    # Top-level Strassen terms
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A01-A11)*(B10+B11)
    # M7 = (A10-A00)*(B00+B01)
    
    terms_A = [
        add_dict(A00, A11),
        add_dict(A10, A11),
        A00,
        A11,
        add_dict(A00, A10),
        sub_dict(A01, A11),
        sub_dict(A10, A00)
    ]
    
    terms_B = [
        add_dict(B00, B11),
        B00,
        sub_dict(B01, B11),
        sub_dict(B10, B00),
        B11,
        add_dict(B10, B11),
        add_dict(B00, B01)
    ]
    
    # Now, for each of the 7 top-level terms, we have a product X_s * Y_s of two 2x2 matrices.
    # X_s is a 2x2 matrix where each entry is a linear combination of A scalars.
    # Y_s is a 2x2 matrix where each entry is a linear combination of B scalars.
    # Wait, X_s and Y_s are the *matrices* themselves, not just blocks.
    # In the top-level Strassen, X_s is a 2x2 matrix of *blocks*.
    # So X_s[0,0] is a 2x2 matrix of A scalars.
    # We need to compute the product of two 2x2 matrices of 2x2 blocks.
    # Let X = [[X00, X01], [X10, X11]] and Y = [[Y00, Y01], [Y10, Y11]].
    # Product P = X Y = [[X00Y00 + X01Y10, X00Y01 + X01Y11], [X10Y00 + X11Y10, X10Y01 + X11Y11]].
    # Each entry of P is a sum of products of 2x2 matrices.
    # Each product of 2x2 matrices (e.g., X00Y00) can be decomposed using rank-7 Strassen.
    
    # However, we can just treat the whole 4x4 product as a sum of 7 terms, each of which is a product of two 2x2 matrices of 2x2 blocks.
    # To get the scalar decomposition, we expand each term.
    
    # Let's define the 2x2 Strassen decomposition for scalar matrices.
    # Given two 2x2 matrices A and B, C = AB.
    # We use 7 products.
    # We will generate the U, V, W entries for these 7 products.
    
    # For a specific top-level term s:
    # X_s is a 2x2 matrix of linear forms in A.
    # Y_s is a 2x2 matrix of linear forms in B.
    # We need to compute the contribution to C.
    # C = sum_s X_s Y_s.
    
    # Let's iterate through the 7 top-level terms.
    for s in range(7):
        X = terms_A[s] # This is a dict of A indices, but wait.
        # terms_A[s] is a single dict representing the *entire* 2x2 block? No.
        # In standard Strassen, M1 = (A00+A11)(B00+B11).
        # Here A00+A11 is a 2x2 matrix.
        # My terms_A[s] is a dict of *scalar* indices? No, A00 is a dict of 4 scalar indices.
        # So terms_A[s] is a dict of 8 scalar indices (if disjoint) or less.
        # This represents the *sum* of two 2x2 blocks, which is a 2x2 matrix of sums.
        # So X_s is a 2x2 matrix where each entry is a linear form.
        # Specifically, X_s[0,0] = A00[0,0] + A11[0,0] = A[0,0] + A[2,2].
        # X_s[0,1] = A00[0,1] + A11[0,1] = A[0,1] + A[2,3].
        # etc.
        
        # Let's construct the 2x2 matrix of linear forms for X_s and Y_s.
        # X_s[i][j] is a dict of (A_idx, coeff).
        
        # We need to map the block structure.
        # A00 is at block (0,0). A11 is at block (1,1).
        # So X_s[0][0] gets contributions from A00[0][0] and A11[0][0].
        
        def get_block_linear_form(term_dict, block_r, block_c, local_r, local_c):
            # term_dict is like A00, which is {(0,0):1, ...} relative to the 4x4 matrix?
            # My A00 dict uses global indices (0,0)..(1,1).
            # So to get the element at local (0,0) of block (0,0), I look up (0,0) in the dict.
            # To get element at local (0,0) of block (1,1), I look up (2,2) in the dict?
            # No, my A11 dict has keys (2,2)..(3,3).
            # So if I want the (0,0) element of the block at (1,1), it corresponds to global (2,2).
            # So I just need to sum the coefficients from the dicts corresponding to the blocks involved.
            pass

        # Let's re-structure.
        # For each top-level term s, we have a set of (block, coeff) pairs for A and B.
        # E.g., M1: A part is {A00: 1, A11: 1}. B part is {B00: 1, B11: 1}.
        # This means X_s = 1*A00 + 1*A11.
        # X_s is a 2x2 matrix of 2x2 blocks.
        # X_s[0][0] = A00[0][0] + A11[0][0].
        
        # Let's define the block coefficients for each term.
        # Term 0: A: {00:1, 11:1}, B: {00:1, 11:1}
        # Term 1: A: {10:1, 11:1}, B: {00:1}
        # Term 2: A: {00:1}, B: {01:1, 11:-1}
        # Term 3: A: {11:1}, B: {10:1, 00:-1}
        # Term 4: A: {00:1, 10:1}, B: {11:1}
        # Term 5: A: {01:1, 11:-1}, B: {10:1, 11:1}
        # Term 6: A: {10:1, 00:-1}, B: {00:1, 01:1}
        
        a_blocks = [
            {(0,0): 1, (1,1): 1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,1): 1},
            {(0,0): 1, (1,0): 1},
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (0,0): -1}
        ]
        
        b_blocks = [
            {(0,0): 1, (1,1): 1},
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (0,0): -1},
            {(1,1): 1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1, (0,1): 1}
        ]
        
        for s in range(7):
            # Construct the 2x2 matrix of linear forms for X_s
            # X_s[i][j] is a dict of (global_A_idx, coeff)
            X = [[{} for _ in range(2)] for _ in range(2)]
            for (br, bc), coeff in a_blocks[s].items():
                # br, bc is the block index (0 or 1)
                # The block covers rows 2*br..2*br+1, cols 2*bc..2*bc+1
                for lr in range(2):
                    for lc in range(2):
                        gr = 2 * br + lr
                        gc = 2 * bc + lc
                        idx = (gr, gc)
                        X[br][bc] # This is wrong. X is the matrix of blocks.
                        # X_s is a 2x2 matrix. The element at (i,j) of X_s is a 2x2 block.
                        # Wait, X_s is the *result* of adding blocks.
                        # X_s[i][j] is the (i,j)-th block of the matrix X_s.
                        # If A part is A00 + A11, then:
                        # X_s[0][0] = A00[0][0] + A11[0][0] ? No.
                        # A00 is the block at (0,0). A11 is the block at (1,1).
                        # So X_s[0][0] gets A00's content. X_s[1][1] gets A11's content.
                        # X_s[0][1] is 0. X_s[1][0] is 0.
                        
                        # So, for each block (br, bc) in the term:
                        # The block content is added to X_s[br][bc].
                        # The block content is a 2x2 matrix of scalars.
                        for lr in range(2):
                            for lc in range(2):
                                gr = 2 * br + lr
                                gc = 2 * bc + lc
                                a_idx_val = gr * 4 + gc
                                if a_idx_val in X[br][bc]:
                                    X[br][bc][a_idx_val] += coeff
                                else:
                                    X[br][bc][a_idx_val] = coeff
                                
            Y = [[{} for _ in range(2)] for _ in range(2)]
            for (br, bc), coeff in b_blocks[s].items():
                for lr in range(2):
                    for lc in range(2):
                        gr = 2 * br + lr
                        gc = 2 * bc + lc
                        b_idx_val = gr * 4 + gc
                        if b_idx_val in Y[br][bc]:
                            Y[br][bc][b_idx_val] += coeff
                        else:
                            Y[br][bc][b_idx_val] = coeff

            # Now we have X and Y as 2x2 matrices of dicts (linear forms).
            # We need to compute P = X Y.
            # P[i][j] = sum_k X[i][k] * Y[k][j].
            # Each term X[i][k] * Y[k][j] is a product of two 2x2 matrices of scalars.
            # We decompose each such product using rank-7 Strassen.
            
            for i in range(2):
                for j in range(2):
                    # P[i][j] is a 2x2 matrix of linear forms in C.
                    # It is the sum of products of 2x2 matrices.
                    # Let's accumulate the linear forms for C[i][j] block.
                    # C_block[i][j] is a 2x2 matrix of scalars.
                    # We need to add the contribution of each product term.
                    
                    # For each k in 0..1:
                    for k in range(2):
                        # Term is X[i][k] * Y[k][j]
                        # X[i][k] is a dict of A indices.
                        # Y[k][j] is a dict of B indices.
                        # This represents a product of two 2x2 matrices (implicitly, the values in the dicts are the entries).
                        # Wait, X[i][k] is a dict of *scalar* indices.
                        # But X[i][k] represents a 2x2 matrix?
                        # No, in my construction above, X[br][bc] accumulated the scalars from the block (br,bc).
                        # So X[0][0] contains the 4 scalars of the top-left block of the matrix X_s.
                        # So X[i][k] is a 2x2 matrix of scalars (represented as a flat dict of 4 entries).
                        
                        # We need to multiply two 2x2 matrices:
                        # A_mat = X[i][k] (2x2)
                        # B_mat = Y[k][j] (2x2)
                        # Result C_mat = A_mat * B_mat (2x2)
                        # We decompose this using 7 Strassen products.
                        
                        # Extract the 2x2 matrices
                        # Indices in dict are global (r, c) -> r*4+c.
                        # Local indices for the 2x2 block at (i, k) of the 4x4 matrix?
                        # No, X[i][k] is the block at row i, col k of the *block matrix*.
                        # The block matrix is 2x2.
                        # The block (i,k) occupies rows 2i..2i+1, cols 2k..2k+1 of the 4x4 matrix.
                        
                        # So the 2x2 matrix A_mat has entries:
                        # A_mat[0][0] = sum of coeffs for global idx (2i, 2k)
                        # A_mat[0][1] = sum of coeffs for global idx (2i, 2k+1)
                        # A_mat[1][0] = sum of coeffs for global idx (2i+1, 2k)
                        # A_mat[1][1] = sum of coeffs for global idx (2i+1, 2k+1)
                        
                        def get_2x2_matrix(block_dict, base_r, base_c):
                            # block_dict is a dict of global_idx -> coeff
                            # We want the 2x2 matrix at rows base_r..base_r+1, cols base_c..base_c+1
                            mat = [[0, 0], [0, 0]]
                            for lr in range(2):
                                for lc in range(2):
                                    gr = base_r + lr
                                    gc = base_c + lc
                                    idx = gr * 4 + gc
                                    if idx in block_dict:
                                        mat[lr][lc] = block_dict[idx]
                            return mat

                        A_mat = get_2x2_matrix(X[i][k], 2*i, 2*k)
                        B_mat = get_2x2_matrix(Y[k][j], 2*k, 2*j)
                        
                        # Now decompose A_mat * B_mat using rank-7 Strassen.
                        # Strassen 2x2:
                        # M1 = (A00+A11)(B00+B11)
                        # M2 = (A10+A11)B00
                        # M3 = A00(B01-B11)
                        # M4 = A11(B10-B00)
                        # M5 = (A00+A10)B11
                        # M6 = (A01-A11)(B10+B11)
                        # M7 = (A10-A00)(B00+B01)
                        
                        # C00 = M1 + M4 - M5 + M7
                        # C01 = M3 + M5
                        # C10 = M2 + M4
                        # C11 = M1 - M2 + M3 + M6
                        
                        # We will add these contributions to the global C matrix.
                        # The result C_mat is at block (i, j) of the block matrix.
                        # So global rows 2i..2i+1, cols 2j..2j+1.
                        
                        # For each of the 7 Strassen products:
                        # We define the linear forms for U (from A_mat) and V (from B_mat).
                        # And the linear form for W (into C_mat).
                        
                        # Let's define the 7 Strassen terms for 2x2 matrices.
                        # Each term t has:
                        # U_t: linear form in A_mat entries
                        # V_t: linear form in B_mat entries
                        # W_t: linear form in C_mat entries (with signs)
                        
                        # We will iterate t=0..6.
                        # For each t, we add a column to U, V, W.
                        
                        for t in range(7):
                            col_idx = s * 7 + t
                            
                            # Determine U_t, V_t, W_t based on t
                            # A_mat entries: A00, A01, A10, A11
                            # B_mat entries: B00, B01, B10, B11
                            
                            # We need to map these to global A and B indices.
                            # A_mat is the block X[i][k].
                            # A00 is at global (2i, 2k).
                            # A01 is at global (2i, 2k+1).
                            # A10 is at global (2i+1, 2k).
                            # A11 is at global (2i+1, 2k+1).
                            
                            a_indices = {
                                'A00': (2*i, 2*k),
                                'A01': (2*i, 2*k+1),
                                'A10': (2*i+1, 2*k),
                                'A11': (2*i+1, 2*k+1)
                            }
                            
                            b_indices = {
                                'B00': (2*k, 2*j),
                                'B01': (2*k, 2*j+1),
                                'B10': (2*k+1, 2*j),
                                'B11': (2*k+1, 2*j+1)
                            }
                            
                            # W indices (C_mat entries)
                            # C_mat is at block (i, j).
                            c_indices = {
                                'C00': (2*i, 2*j),
                                'C01': (2*i, 2*j+1),
                                'C10': (2*i+1, 2*j),
                                'C11': (2*i+1, 2*j+1)
                            }
                            
                            # Define the linear forms for U, V, W for each t
                            # t=0: M1
                            if t == 0:
                                u_form = {'A00': 1, 'A11': 1}
                                v_form = {'B00': 1, 'B11': 1}
                                w_form = {'C00': 1, 'C11': 1}
                            elif t == 1:
                                u_form = {'A10': 1, 'A11': 1}
                                v_form = {'B00': 1}
                                w_form = {'C10': 1, 'C11': -1}
                            elif t == 2:
                                u_form = {'A00': 1}
                                v_form = {'B01': 1, 'B11': -1}
                                w_form = {'C01': 1, 'C11': 1}
                            elif t == 3:
                                u_form = {'A11': 1}
                                v_form = {'B10': 1, 'B00': -1}
                                w_form = {'C00': 1, 'C10': 1}
                            elif t == 4:
                                u_form = {'A00': 1, 'A10': 1}
                                v_form = {'B11': 1}
                                w_form = {'C00': -1, 'C11': 1}
                            elif t == 5:
                                u_form = {'A01': 1, 'A11': -1}
                                v_form = {'B10': 1, 'B11': 1}
                                w_form = {'C11': 1}
                            elif t == 6:
                                u_form = {'A10': 1, 'A00': -1}
                                v_form = {'B00': 1, 'B01': 1}
                                w_form = {'C00': 1, 'C10': -1}
                                
                            # Add to U, V, W matrices
                            for name, coeff in u_form.items():
                                r, c = a_indices[name]
                                U[r * 4 + c][col_idx] += coeff
                                
                            for name, coeff in v_form.items():
                                r, c = b_indices[name]
                                V[r * 4 + c][col_idx] += coeff
                                
                            for name, coeff in w_form.items():
                                r, c = c_indices[name]
                                W[r * 4 + c][col_idx] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
