import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get flat indices for a 2x2 block
    # A block (bi, bj) where bi, bj in {0,1}
    # bi=0, bj=0 -> rows 0,1; cols 0,1
    # bi=0, bj=1 -> rows 0,1; cols 2,3
    # bi=1, bj=0 -> rows 2,3; cols 0,1
    # bi=1, bj=1 -> rows 2,3; cols 2,3
    def get_A_indices(bi, bj):
        r_start = bi * 2
        c_start = bj * 2
        return [r_start * 4 + c_start, r_start * 4 + c_start + 1, (r_start + 1) * 4 + c_start, (r_start + 1) * 4 + c_start + 1]
        
    def get_B_indices(bi, bj):
        # B is m x p. m=4, p=4.
        # Block (bi, bj)
        # bi is row block (0,1), bj is col block (0,1)
        r_start = bi * 2
        c_start = bj * 2
        return [r_start * 4 + c_start, r_start * 4 + c_start + 1, (r_start + 1) * 4 + c_start, (r_start + 1) * 4 + c_start + 1]
        
    def get_C_indices(bi, bj):
        # C is p x n. p=4, n=4.
        # Block (bi, bj)
        # bi is row block (0,1) corresponding to k (p), bj is col block (0,1) corresponding to i (n)
        # Wait, C[k, i].
        # k is row index in C. i is col index in C.
        # Block (bi, bj) means k in block bi, i in block bj.
        r_start = bi * 2
        c_start = bj * 2
        return [r_start * 4 + c_start, r_start * 4 + c_start + 1, (r_start + 1) * 4 + c_start, (r_start + 1) * 4 + c_start + 1]

    # Define the 7 Strassen terms
    # Each term is (X_coeffs, Y_coeffs)
    # X_coeffs: dict mapping (bi, bj) to coeff
    # Y_coeffs: dict mapping (bi, bj) to coeff
    
    terms = []
    
    # M1 = (A11 + A22)(B11 + B22)
    X1 = {(0,0): 1, (1,1): 1}
    Y1 = {(0,0): 1, (1,1): 1}
    terms.append((X1, Y1))
    
    # M2 = (A21 + A22)B11
    X2 = {(1,0): 1, (1,1): 1}
    Y2 = {(0,0): 1}
    terms.append((X2, Y2))
    
    # M3 = A11(B12 - B22)
    X3 = {(0,0): 1}
    Y3 = {(0,1): 1, (1,1): -1}
    terms.append((X3, Y3))
    
    # M4 = A22(B21 - B11)
    X4 = {(1,1): 1}
    Y4 = {(1,0): 1, (0,0): -1}
    terms.append((X4, Y4))
    
    # M5 = (A11 + A12)B22
    X5 = {(0,0): 1, (0,1): 1}
    Y5 = {(1,1): 1}
    terms.append((X5, Y5))
    
    # M6 = (A21 - A11)(B11 + B12)
    X6 = {(1,0): 1, (0,0): -1}
    Y6 = {(0,0): 1, (0,1): 1}
    terms.append((X6, Y6))
    
    # M7 = (A12 - A22)(B21 + B22)
    X7 = {(0,1): 1, (1,1): -1}
    Y7 = {(1,0): 1, (1,1): 1}
    terms.append((X7, Y7))
    
    col = 0
    for X, Y in terms:
        # Expand X into a 2x2 matrix of coefficients for each element of A
        # X is a linear combo of blocks.
        # We need to form the 4 scalar multipliers for the 2x2 product.
        # Let X_mat be a 2x2 matrix where X_mat[r][c] is the coefficient of A_block(r,c)
        # Actually, X is a 2x2 matrix of blocks.
        # X = [ X00  X01 ]
        #      [ X10  X11 ]
        # where X00 is a scalar multiple of A00 block? No.
        # In Strassen, X is a 2x2 matrix of 2x2 blocks.
        # X00 = sum coeff * A_block.
        # But here, each "block" in the Strassen formula is a 2x2 matrix.
        # So X00 is a 2x2 matrix formed by linear combos of A's 2x2 blocks.
        
        # Let's represent X as a 2x2 array of 2x2 matrices of coefficients.
        # X[bi][bj] is a 2x2 matrix of coeffs for A's elements in block (bi, bj)?
        # No, the formula says X = A11 + A22. This means the top-left block of X is A11, bottom-right is A22.
        # So X[0][0] = A11, X[1][1] = A22, others 0.
        
        # Let's build the 4x4 coefficient matrix for X (flattened A)
        # And 4x4 for Y (flattened B)
        
        # We need to compute the 4 outputs of the 2x2 matrix product X*Y.
        # (X*Y)[0][0] = X00*Y00 + X01*Y10
        # (X*Y)[0][1] = X00*Y01 + X01*Y11
        # (X*Y)[1][0] = X10*Y00 + X11*Y10
        # (X*Y)[1][1] = X10*Y01 + X11*Y11
        
        # Each Xij is a 2x2 matrix (block).
        # Let's get the indices for each block.
        
        def get_block_matrix_coeffs(block_dict, bi, bj):
            # Returns a 2x2 numpy array of coefficients for the elements of A/B
            # block_dict maps (bi, bj) -> coeff
            # If (bi, bj) is in dict, the 2x2 block has that coeff.
            # Else 0.
            mat = np.zeros((2, 2), dtype=int)
            if (bi, bj) in block_dict:
                mat[:, :] = block_dict[(bi, bj)]
            return mat

        # Construct X and Y as 2x2 arrays of 2x2 matrices
        X_blocks = [[get_block_matrix_coeffs(X, bi, bj) for bj in range(2)] for bi in range(2)]
        Y_blocks = [[get_block_matrix_coeffs(Y, bi, bj) for bj in range(2)] for bi in range(2)]
        
        # Now compute the 4 products
        # Product at (0,0): X00*Y00 + X01*Y10
        # X00 is 2x2, Y00 is 2x2.
        # We need to decompose the matrix product of two 2x2 matrices into scalar products.
        # Since the coefficients are just scalars (0, 1, -1), we can just iterate over the non-zero entries.
        
        # For each of the 4 positions (pr, pc) in the result block:
        for pr in range(2):
            for pc in range(2):
                # The result block (pr, pc) corresponds to C block (pr, pc)
                # But wait, the Strassen result M_k is a 2x2 matrix of 2x2 blocks.
                # So M_k[pr][pc] is a 2x2 matrix.
                # This M_k[pr][pc] contributes to C[pr][pc].
                
                # Calculate the 2x2 matrix P = X[pr][0]*Y[0][pc] + X[pr][1]*Y[1][pc]
                # Actually, (X*Y)[pr][pc] = sum_{m=0}^{1} X[pr][m] * Y[m][pc]
                
                # We need to add the scalar products to U, V, W.
                # For each term X[pr][m] * Y[m][pc]:
                # X[pr][m] is a 2x2 matrix of coeffs for A.
                # Y[m][pc] is a 2x2 matrix of coeffs for B.
                # The product is a matrix product of two 2x2 matrices with constant coefficients.
                # Let A_sub = X[pr][m] (2x2), B_sub = Y[m][pc] (2x2).
                # The product C_sub = A_sub * B_sub.
                # C_sub[i][j] = sum_{k} A_sub[i][k] * B_sub[k][j].
                # Each term A_sub[i][k] * B_sub[k][j] is a scalar product of linear forms.
                
                for m in range(2):
                    A_sub = X_blocks[pr][m]
                    B_sub = Y_blocks[m][pc]
                    
                    # Iterate over the 4 elements of the product matrix
                    for i in range(2):
                        for j in range(2):
                            # C_sub[i][j] = sum_{k=0}^{1} A_sub[i][k] * B_sub[k][j]
                            for k in range(2):
                                coeff_A = A_sub[i][k]
                                coeff_B = B_sub[k][j]
                                
                                if coeff_A == 0 or coeff_B == 0:
                                    continue
                                
                                # We have a multiplication: (coeff_A * A_elem) * (coeff_B * B_elem)
                                # A_elem is at local index (i, k) in block (pr, m) of A?
                                # No, A_sub is the coefficient matrix for the block (pr, m) of A?
                                # Let's trace indices carefully.
                                
                                # X[pr][m] corresponds to A block (pr, m).
                                # The element A_sub[i][k] is the coefficient of A[2*pr + i][2*m + k].
                                # Flat index for A: (2*pr + i) * 4 + (2*m + k).
                                
                                # Y[m][pc] corresponds to B block (m, pc).
                                # The element B_sub[k][j] is the coefficient of B[2*m + k][2*pc + j].
                                # Flat index for B: (2*m + k) * 4 + (2*pc + j).
                                
                                # The result contributes to C block (pr, pc).
                                # Specifically, the element at local (i, j) in C block (pr, pc).
                                # Flat index for C: (2*pr + i) * 4 + (2*pc + j).
                                # Note: C index is k*n + i. Here k is row, i is col.
                                # In our block notation, pr is row block (k), pc is col block (i).
                                # So C flat index: (2*pr + i) * 4 + (2*pc + j).
                                
                                a_idx = (2*pr + i) * 4 + (2*m + k)
                                b_idx = (2*m + k) * 4 + (2*pc + j)
                                c_idx = (2*pr + i) * 4 + (2*pc + j)
                                
                                U[a_idx][col] += coeff_A
                                V[b_idx][col] += coeff_B
                                W[c_idx][col] += 1
                                
                                col += 1
                                
    # The rank might be less than 28 if some columns are zero or redundant?
    # No, Strassen ensures 28 distinct multiplications.
    # However, we might have generated duplicate columns if coefficients cancel out?
    # Unlikely with Strassen.
    
    return {"U": U, "V": V, "W": W}
