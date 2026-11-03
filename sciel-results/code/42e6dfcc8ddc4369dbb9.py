"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-49 decomposition of the 4x4 matrix multiplication tensor
    using Strassen's algorithm.
    """
    n, m, p = DIMENSIONS
    rank = 49
    
    # U, V, W will be lists of lists to match the expected interface
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get block indices
    # A is 4x4. Blocks are 2x2.
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # We represent linear forms as dicts or arrays of size 16.
    # Let's use arrays of size 16 for A and B.
    
    # Define the 7 Strassen products for 2x2 matrices
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # For each s in 1..7, we define X_s and Y_s as 2x2 matrices of linear forms.
    # X_s and Y_s are 2x2 arrays where each element is a linear form (array of 16).
    
    # Let's define the blocks of A as vectors of length 16.
    # A11: indices 0,1,4,5
    # A12: indices 2,3,6,7
    # A21: indices 8,9,12,13
    # A22: indices 10,11,14,15
    
    def get_block_indices(block_name):
        if block_name == 'A11': return [0, 1, 4, 5]
        if block_name == 'A12': return [2, 3, 6, 7]
        if block_name == 'A21': return [8, 9, 12, 13]
        if block_name == 'A22': return [10, 11, 14, 15]
        # Same for B
        if block_name == 'B11': return [0, 1, 4, 5]
        if block_name == 'B12': return [2, 3, 6, 7]
        if block_name == 'B21': return [8, 9, 12, 13]
        if block_name == 'B22': return [10, 11, 14, 15]

    # We will build U, V, W column by column.
    # Each column corresponds to a term (s, t).
    
    col = 0
    for s in range(1, 8):
        # Define X_s and Y_s
        # X_s is 2x2, Y_s is 2x2.
        # Each entry is a linear form in A or B.
        
        # Initialize X_s and Y_s as 2x2 arrays of zeros (length 16)
        X_s = [[np.zeros(16) for _ in range(2)] for _ in range(2)]
        Y_s = [[np.zeros(16) for _ in range(2)] for _ in range(2)]
        
        # Populate X_s and Y_s based on s
        # s=1: X1 = A11 + A22, Y1 = B11 + B22
        # s=2: X2 = A21 + A22, Y2 = B11
        # s=3: X3 = A11, Y3 = B12 - B22
        # s=4: X4 = A22, Y4 = B21 - B11
        # s=5: X5 = A11 + A12, Y5 = B22
        # s=6: X6 = A21 - A11, Y6 = B11 + B12
        # s=7: X7 = A12 - A22, Y7 = B21 + B22
        
        def add_block(X, block, coeff, is_A=True):
            # block is like 'A11'
            prefix = 'A' if is_A else 'B'
            block_name = prefix + block[1:]
            indices = get_block_indices(block_name)
            for idx in indices:
                X[idx] += coeff

        # This helper is for a single 2x2 matrix.
        # Let's just manually construct X_s and Y_s entries.
        
        # X_s entries:
        # X_s[0][0], X_s[0][1], X_s[1][0], X_s[1][1]
        
        # For each s, we define the 4 entries of X_s and Y_s.
        # Each entry is a sum of blocks.
        
        # s=1
        if s == 1:
            # X1 = A11 + A22 (same for all entries? No, X1 is a matrix)
            # M1 = (A11 + A22)(B11 + B22)
            # This means X1 = A11 + A22 and Y1 = B11 + B22?
            # Wait, (A11 + A22) is a 2x2 matrix.
            # So X1[0][0] = A11[0][0] + A22[0][0], etc.
            # So X1 is the matrix A11 + A22.
            # Y1 is the matrix B11 + B22.
            
            # Let's define a function to set X_s to a sum of blocks.
            def set_X_s(X, blocks, coeffs):
                # blocks: list of block names
                # coeffs: list of coefficients
                for b, c in zip(blocks, coeffs):
                    indices = get_block_indices(b)
                    for idx in indices:
                        # We need to map the flat index to 2x2 position in the block
                        # But X_s is 2x2 of linear forms.
                        # The linear form is over the whole 4x4 matrix.
                        # So we just add to the global linear form.
                        # But X_s is a 2x2 matrix of linear forms?
                        # No, X_s is a 2x2 matrix where each entry is a linear form.
                        # The product M_s = X_s Y_s is a 2x2 matrix.
                        # M_s[i][j] = sum_k X_s[i][k] Y_s[k][j].
                        # Each X_s[i][k] is a scalar linear form?
                        # No, X_s is a 2x2 matrix of 2x2 matrices?
                        # No, Strassen's algorithm operates on blocks.
                        # A11 is a 2x2 matrix.
                        # X1 = A11 + A22 is a 2x2 matrix.
                        # So X1[i][j] is the (i,j)-th element of the 2x2 matrix A11 + A22.
                        # This element is a linear combination of elements of A.
                        
                        # Let's refine the representation.
                        # X_s is a 2x2 array. Each element X_s[i][j] is a linear form (array of 16).
                        # The linear form corresponds to the (i,j)-th element of the block matrix X_s.
                        
                        pass # Placeholder

            # Let's implement the logic directly for each s.
            
            # X1 = A11 + A22
            # X1[0][0] = A[0][0] + A[2][2]
            # X1[0][1] = A[0][1] + A[2][3]
            # X1[1][0] = A[1][0] + A[3][2]
            # X1[1][1] = A[1][1] + A[3][3]
            
            # We can define the indices for each entry of the block matrix.
            # Block A11:
            # (0,0) -> 0, (0,1) -> 1, (1,0) -> 4, (1,1) -> 5
            # Block A22:
            # (0,0) -> 10, (0,1) -> 11, (1,0) -> 14, (1,1) -> 15
            
            # Let's create a helper to get indices for a specific entry of a block.
            def get_block_entry_indices(block_name, r, c):
                # r, c in 0..1
                base = {'A11': 0, 'A12': 2, 'A21': 8, 'A22': 10,
                        'B11': 0, 'B12': 2, 'B21': 8, 'B22': 10}
                # Wait, B indices are the same as A for the linear form V.
                # V is 16x49.
                return base[block_name] + r * 2 + c

            # Now populate X_s and Y_s.
            
            if s == 1:
                # X1 = A11 + A22
                for r in range(2):
                    for c in range(2):
                        idx1 = get_block_entry_indices('A11', r, c)
                        idx2 = get_block_entry_indices('A22', r, c)
                        X_s[r][c][idx1] = 1
                        X_s[r][c][idx2] = 1
                        # Y1 = B11 + B22
                        idx3 = get_block_entry_indices('B11', r, c)
                        idx4 = get_block_entry_indices('B22', r, c)
                        Y_s[r][c][idx3] = 1
                        Y_s[r][c][idx4] = 1
                        
            elif s == 2:
                # X2 = A21 + A22
                for r in range(2):
                    for c in range(2):
                        idx1 = get_block_entry_indices('A21', r, c)
                        idx2 = get_block_entry_indices('A22', r, c)
                        X_s[r][c][idx1] = 1
                        X_s[r][c][idx2] = 1
                        # Y2 = B11
                        idx3 = get_block_entry_indices('B11', r, c)
                        Y_s[r][c][idx3] = 1

            elif s == 3:
                # X3 = A11
                for r in range(2):
                    for c in range(2):
                        idx1 = get_block_entry_indices('A11', r, c)
                        X_s[r][c][idx1] = 1
                        # Y3 = B12 - B22
                        idx2 = get_block_entry_indices('B12', r, c)
                        idx3 = get_block_entry_indices('B22', r, c)
                        Y_s[r][c][idx2] = 1
                        Y_s[r][c][idx3] = -1

            elif s == 4:
                # X4 = A22
                for r in range(2):
                    for c in range(2):
                        idx1 = get_block_entry_indices('A22', r, c)
                        X_s[r][c][idx1] = 1
                        # Y4 = B21 - B11
                        idx2 = get_block_entry_indices('B21', r, c)
                        idx3 = get_block_entry_indices('B11', r, c)
                        Y_s[r][c][idx2] = 1
                        Y_s[r][c][idx3] = -1

            elif s == 5:
                # X5 = A11 + A12
                for r in range(2):
                    for c in range(2):
                        idx1 = get_block_entry_indices('A11', r, c)
                        idx2 = get_block_entry_indices('A12', r, c)
                        X_s[r][c][idx1] = 1
                        X_s[r][c][idx2] = 1
                        # Y5 = B22
                        idx3 = get_block_entry_indices('B22', r, c)
                        Y_s[r][c][idx3] = 1

            elif s == 6:
                # X6 = A21 - A11
                for r in range(2):
                    for c in range(2):
                        idx1 = get_block_entry_indices('A21', r, c)
                        idx2 = get_block_entry_indices('A11', r, c)
                        X_s[r][c][idx1] = 1
                        X_s[r][c][idx2] = -1
                        # Y6 = B11 + B12
                        idx3 = get_block_entry_indices('B11', r, c)
                        idx4 = get_block_entry_indices('B12', r, c)
                        Y_s[r][c][idx3] = 1
                        Y_s[r][c][idx4] = 1

            elif s == 7:
                # X7 = A12 - A22
                for r in range(2):
                    for c in range(2):
                        idx1 = get_block_entry_indices('A12', r, c)
                        idx2 = get_block_entry_indices('A22', r, c)
                        X_s[r][c][idx1] = 1
                        X_s[r][c][idx2] = -1
                        # Y7 = B21 + B22
                        idx3 = get_block_entry_indices('B21', r, c)
                        idx4 = get_block_entry_indices('B22', r, c)
                        Y_s[r][c][idx3] = 1
                        Y_s[r][c][idx4] = 1

        # Now we have X_s and Y_s.
        # We need to compute the 7 products P_{s,t} for M_s = X_s Y_s.
        # M_s[i][j] = sum_k X_s[i][k] Y_s[k][j]
        
        # The 7 Strassen products for M_s:
        # P1 = (X00 + X11)(Y00 + Y11)
        # P2 = (X10 + X11)Y00
        # P3 = X00(Y01 - Y11)
        # P4 = X11(Y10 - Y00)
        # P5 = (X00 + X01)Y11
        # P6 = (X10 - X00)(Y00 + Y01)
        # P7 = (X01 - X11)(Y10 + Y11)
        
        # We need to expand these into linear forms L_{s,t}(A) and R_{s,t}(B).
        # Then find the contribution to C.
        
        # Let's store the linear forms for each P_{s,t}.
        # L_{s,t} is an array of 16.
        # R_{s,t} is an array of 16.
        
        for t in range(1, 8):
            L = np.zeros(16)
            R = np.zeros(16)
            
            if t == 1:
                # (X00 + X11)(Y00 + Y11)
                # L = X00 + X11
                L = X_s[0][0] + X_s[1][1]
                # R = Y00 + Y11
                R = Y_s[0][0] + Y_s[1][1]
                
            elif t == 2:
                # (X10 + X11)Y00
                L = X_s[1][0] + X_s[1][1]
                R = Y_s[0][0]
                
            elif t == 3:
                # X00(Y01 - Y11)
                L = X_s[0][0]
                R = Y_s[0][1] - Y_s[1][1]
                
            elif t == 4:
                # X11(Y10 - Y00)
                L = X_s[1][1]
                R = Y_s[1][0] - Y_s[0][0]
                
            elif t == 5:
                # (X00 + X01)Y11
                L = X_s[0][0] + X_s[0][1]
                R = Y_s[1][1]
                
            elif t == 6:
                # (X10 - X00)(Y00 + Y01)
                L = X_s[1][0] - X_s[0][0]
                R = Y_s[0][0] + Y_s[0][1]
                
            elif t == 7:
                # (X01 - X11)(Y10 + Y11)
                L = X_s[0][1] - X_s[1][1]
                R = Y_s[1][0] + Y_s[1][1]
            
            # Now we have L and R.
            # We need to determine the contribution of P_{s,t} to C.
            # M_s[i][j] is a linear combination of P_{s,t}.
            # Coefficients:
            # M00 = P1 + P4 - P5 + P7
            # M01 = P3 + P5
            # M10 = P2 + P4
            # M11 = P1 - P2 + P3 + P6
            
            coeff_in_M = 0
            if t == 1:
                coeff_in_M_00 = 1
                coeff_in_M_01 = 0
                coeff_in_M_10 = 0
                coeff_in_M_11 = 1
            elif t == 2:
                coeff_in_M_00 = 0
                coeff_in_M_01 = 0
                coeff_in_M_10 = 1
                coeff_in_M_11 = -1
            elif t == 3:
                coeff_in_M_00 = 0
                coeff_in_M_01 = 1
                coeff_in_M_10 = 0
                coeff_in_M_11 = 1
            elif t == 4:
                coeff_in_M_00 = 1
                coeff_in_M_01 = 0
                coeff_in_M_10 = 1
                coeff_in_M_11 = 0
            elif t == 5:
                coeff_in_M_00 = -1
                coeff_in_M_01 = 1
                coeff_in_M_10 = 0
                coeff_in_M_11 = 0
            elif t == 6:
                coeff_in_M_00 = 0
                coeff_in_M_01 = 0
                coeff_in_M_10 = 0
                coeff_in_M_11 = 1
            elif t == 7:
                coeff_in_M_00 = 1
                coeff_in_M_01 = 0
                coeff_in_M_10 = 0
                coeff_in_M_11 = 0
            
            # Now, M_s contributes to C blocks.
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # M_s is a 2x2 matrix.
            # M_s[0][0] contributes to C[0][0] (top-left 2x2 block, top-left element)
            # M_s[0][1] contributes to C[0][1] (top-left 2x2 block, top-right element)
            # M_s[1][0] contributes to C[1][0] (top-left 2x2 block, bottom-left element)
            # M_s[1][1] contributes to C[1][1] (top-left 2x2 block, bottom-right element)
            
            # Wait, C is 4x4.
            # C00 is the top-left 2x2 block of C.
            # So M_s[0][0] contributes to C[0][0].
            # M_s[0][1] contributes to C[0][1].
            # M_s[1][0] contributes to C[1][0].
            # M_s[1][1] contributes to C[1][1].
            
            # But M_s is multiplied by coefficients from the C block assembly.
            
            # Let's define the coefficients of M_s in C blocks.
            # C_block_coeffs[s] is a 2x2 matrix of coefficients.
            
            if s == 1:
                # C00 += M1, C11 += M1
                c_coeff_00 = 1 # for C00 block
                c_coeff_01 = 0
                c_coeff_10 = 0
                c_coeff_11 = 1 # for C11 block
            elif s == 2:
                c_coeff_00 = 0
                c_coeff_01 = 0
                c_coeff_10 = 1 # C10 += M2
                c_coeff_11 = -1 # C11 -= M2
            elif s == 3:
                c_coeff_00 = 0
                c_coeff_01 = 1 # C01 += M3
                c_coeff_10 = 0
                c_coeff_11 = 1 # C11 += M3
            elif s == 4:
                c_coeff_00 = 1 # C00 += M4
                c_coeff_01 = 0
                c_coeff_10 = 1 # C10 += M4
                c_coeff_11 = 0
            elif s == 5:
                c_coeff_00 = -1 # C00 -= M5
                c_coeff_01 = 1 # C01 += M5
                c_coeff_10 = 0
                c_coeff_11 = 0
            elif s == 6:
                c_coeff_00 = 0
                c_coeff_01 = 0
                c_coeff_10 = 0
                c_coeff_11 = 1 # C11 += M6
            elif s == 7:
                c_coeff_00 = 1 # C00 += M7
                c_coeff_01 = 0
                c_coeff_10 = 0
                c_coeff_11 = 0

            # Now, for each entry (i,j) of M_s, the total coefficient in C is:
            # coeff_C[i][j] = coeff_in_M[i][j] * c_coeff[i][j]
            
            # We need to map this to the W matrix.
            # W index is 4k + i.
            # C[i][j] corresponds to i_global, j_global.
            # C block 00: i_global in 0..1, j_global in 0..1.
            # C block 01: i_global in 0..1, j_global in 2..3.
            # C block 10: i_global in 2..3, j_global in 0..1.
            # C block 11: i_global in 2..3, j_global in 2..3.
            
            # For a given s, t, we have L (for U) and R (for V).
            # We need to fill U[:, col], V[:, col], W[:, col].
            
            # U is 16x49.
            # V is 16x49.
            # W is 16x49.
            
            # Fill U and V with L and R.
            for idx in range(16):
                U[idx][col] = L[idx]
                V[idx][col] = R[idx]
            
            # Fill W.
            # For each i_global, j_global, the coefficient is:
            # if (i_global, j_global) is in block (bi, bj):
            #   coeff = coeff_in_M[bi][bj] * c_coeff[bi][bj]
            #   W[4*j_global + i_global][col] = coeff
            
            for bi in range(2):
                for bj in range(2):
                    # Get coeff_in_M for this block
                    if bi == 0 and bj == 0:
                        c_im = coeff_in_M_00
                    elif bi == 0 and bj == 1:
                        c_im = coeff_in_M_01
                    elif bi == 1 and bj == 0:
                        c_im = coeff_in_M_10
                    else:
                        c_im = coeff_in_M_11
                    
                    # Get c_coeff for this block
                    if bi == 0 and bj == 0:
                        c_cc = c_coeff_00
                    elif bi == 0 and bj == 1:
                        c_cc = c_coeff_01
                    elif bi == 1 and bj == 0:
                        c_cc = c_coeff_10
                    else:
                        c_cc = c_coeff_11
                    
                    total_coeff = c_im * c_cc
                    
                    if total_coeff != 0:
                        # Map to global indices
                        # Block (bi, bj)
                        # i_global = bi * 2 + i_local
                        # j_global = bj * 2 + j_local
                        for i_local in range(2):
                            for j_local in range(2):
                                i_global = bi * 2 + i_local
                                j_global = bj * 2 + j_local
                                w_idx = 4 * j_global + i_global
                                W[w_idx][col] = total_coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
