"""
Matrix multiplication 4x4x8 tensor decomposition.
Uses Strassen-Schoolbook: rank 7 for 4x4x4 core, times 2 for the 8-wide second dim = 56.
Score = 128/56 = 2.2857...
"""

def construct():
    n, m, p = 4, 4, 8
    
    # Each 4x4x4 block decomposed via Strassen (7 rank-1 terms at 4x4 level,
    # but each "rank-1" at the top level is itself a 2x2x2 tensor = 4 rank-1 terms).
    # Total per 4x4x4 block: 7 * 4 = 28 rank-1 terms.
    # Two blocks: 2 * 28 = 56.
    
    rank = 56
    
    # Build U (n*m = 16 rows), V (m*p = 32 rows), W (p*n = 32 rows)
    # Convention: output index is k*n + i (row k of A times column i of B? 
    # Actually W[k*n+i, r] means output element at position (i, k) in the result 
    # is sum_r U[i, r] * V[k, r]... wait let me re-read.
    
    # The contract says: W[k*n+i, r]
    # For multiplying A (n x m) by B (m x p), result C (n x p), C[i, k] = sum_j A[i,j]*B[j,k]
    # The trace convention W[k*n+i, r] means the output coordinate (i,k) maps to row k*n+i of W.
    # U has rows indexed by (i, j) -> row i*m+j, so U[i*m+j, r] corresponds to A[i,j]
    # V has rows indexed by (j, k) -> row j*p+k, so V[j*p+k, r] corresponds to B[j,k]
    # W[k*n+i, r] corresponds to C[i,k]
    
    # So: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j as well?
    # No. The tensor decomposition means:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]  ... but the indices need to match.
    
    # Actually for matrix multiplication A*B where A is n×m and B is m×p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The multiplication tensor has indices (i, j, k) with the j contracted.
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    
    # In tensor decomposition form:
    # C[i,k] = sum_r (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k]) * W[k,i,r]
    # No, that's not right either. Let me think more carefully.
    
    # Standard: The matrix multiplication tensor M_{n,m,p} has entries M[i,j,k] = 1 
    # representing A[i,j]*B[j,k] -> C[i,k].
    # A rank-1 decomposition: M = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r has entries u_r[i,j], v_r has entries v_r[j,k], w_r has entries w_r[i,k]
    # Such that for all i,j,k: M[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[i,k]
    # But M[i,j,k] = 1 for all valid i,j,k (it's just the structure).
    
    # Wait, that's not quite right. The tensor is:
    # T[i,j,k] = 1  (meaning the product A[i,j]*B[j,k] appears in C[i,k])
    # And the decomposition satisfies:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i,j,k
    
    # No wait. Let me re-read the schoolbook:
    # For each (i,j,k): U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
    # So sum over columns: for a given (i,j,k), only the column corresponding to (i,j,k) has all three = 1.
    # So sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for each (i,j,k), and 0 otherwise.
    
    # This means the bilinear form is:
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] * A[i,j]*B[j,k]
    # Hmm, this is getting confusing. Let me just follow the schoolbook pattern.
    
    # For Strassen on 4x4: split A into 2x2 blocks, B into 2x2 blocks, C into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]]  (each 2x2)
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mx is a 2x2 matrix product, which requires 4 scalar multiplications (schoolbook).
    # So total: 7 * 4 = 28 rank-1 terms per 4x4x4 block.
    
    # For the 4x4x8 problem, B is 4x8, so split into two 4x4 blocks: B = [B_left, B_right]
    # C = [C_left, C_right] where C_left = A*B_left, C_right = A*B_right
    # Each is a 4x4x4 multiplication, so total rank = 2 * 28 = 56.
    
    # Now I need to construct U, V, W explicitly.
    
    # Let me define the 2x2 schoolbook first as a helper concept, then build up.
    
    # For a 2x2x2 multiplication (n=m=p=2):
    # U rows: i*m+j for i in 0..1, j in 0..1 -> rows 0,1,2,3
    # V rows: j*p+k for j in 0..1, k in 0..1 -> rows 0,1,2,3  
    # W rows: k*n+i for k in 0..1, i in 0..1 -> rows 0,1,2,3
    # Rank 4, one term per (i,j,k).
    
    # For Strassen 4x4x4:
    # I'll construct it by defining the 7 intermediate products and their schoolbook decompositions.
    
    # Let me build this step by step.
    
    # First, let me define a function to create a schoolbook decomposition for 2x2x2
    # with a given offset for the rows/columns and a sign.
    
    # Actually, let me just directly construct the 56-term decomposition.
    
    # For each of the 7 Strassen products, each involves linear combinations of 2x2 submatrices.
    # Let me think of it as: for each Strassen term, I have a 2x2x2 multiplication with 
    # specific linear combinations in U and V.
    
    # Let me index the 4x4 matrices by (block_row, block_col) where each is 2x2.
    # A[i,j] where i = 2*ar + a, j = 2*aj + b (ar,aj in 0,1; a,b in 0,1)
    # Similarly for B and C.
    
    # This is getting complex. Let me just hard-code the 56 terms.
    
    # Alternative: Let me build it programmatically.
    
    # For the 4x4x4 Strassen decomposition:
    # The 7 products are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A11+A12-A22)(B21+B22)  -- wait, let me double-check Strassen
    
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Now, each Mx is a 2x2 matrix, computed as a product of two 2x2 matrices (which may be sums/differences of subblocks).
    
    # For the tensor decomposition, I need to express this as rank-1 terms.
    
    # Let me create the U, V, W matrices directly.
    
    # U has 16 rows (4x4 for A), V has 32 rows (4x8 for B), W has 32 rows (8x4 for C output, indexed as k*n+i)
    
    # For the first 4x4x4 block (B_left, C_left):
    # Rows of U: 0-15 (A indices)
    # Rows of V: 0-15 (B_left indices: j*p+k where j in 0..3, k in 0..3)
    # Rows of W: 0-15 (C_left indices: k*n+i where k in 0..3, i in 0..3)
    
    # For the second 4x4x4 block (B_right, C_right):
    # Rows of U: 0-15 (same A)
    # Rows of V: 16-31 (B_right indices: j*p+k where j in 0..3, k in 4..7, so row j*8+(4+k'))
    # Rows of W: 16-31 (C_right indices: k*n+i where k in 4..7, i in 0..3, so row (4+k')*4+i)
    
    # Let me build one 4x4x4 Strassen decomposition first, then duplicate it.
    
    # For one 4x4x4 block, I'll create U_block (16x28), V_block (16x28), W_block (16x28)
    
    # The 7 Strassen products, each is a 2x2x2 schoolbook (4 terms):
    
    # Let me define the subblock indices:
    # A11: rows 0-1, cols 0-1 -> U rows 0,1,4,5  (i*m+j: i=0,j=0 ->0; i=0,j=1->1; i=1,j=0->4; i=1,j=1->5)
    # A12: rows 0-1, cols 2-3 -> U rows 2,3,6,7
    # A21: rows 2-3, cols 0-1 -> U rows 8,9,12,13
    # A22: rows 2-3, cols 2-3 -> U rows 10,11,14,15
    
    # B11: rows 0-1, cols 0-1 -> V rows 0,1,8,9  (j*p+k: j=0,k=0->0; j=0,k=1->1; j=1,k=0->8; j=1,k=1->9)
    # B12: rows 0-1, cols 2-3 -> V rows 2,3,10,11
    # B21: rows 2-3, cols 0-1 -> V rows 16,17,24,25  -- wait, for 4x4 B: j*p+k, p=4
    # j=2,k=0 -> 2*4+0=8; j=2,k=1->9; j=3,k=0->12; j=3,k=1->13
    # B21: rows 2-3, cols 0-1 -> V rows 8,9,12,13
    # B22: rows 2-3, cols 2-3 -> V rows 10,11,14,15
    
    # C11: k=0-1, i=0-1 -> W rows: k*n+i: k=0,i=0->0; k=0,i=1->1; k=1,i=0->4; k=1,i=1->5
    # C12: k=2-3, i=0-1 -> W rows: k=2,i=0->8; k=2,i=1->9; k=3,i=0->12; k=3,i=1->13
    # C21: k=0-1, i=2-3 -> W rows: k=0,i=2->2; k=0,i=3->3; k=1,i=2->6; k=1,i=3->7
    # C22: k=2-3, i=2-3 -> W rows: k=2,i=2->10; k=2,i=3->11; k=3,i=2->14; k=3,i=3->15
    
    # Now for each Strassen term Mx, it's a 2x2 product. Let me define the linear combos.
    
    # I'll create a helper that given linear coefficients for the 4 subblocks of A-side and B-side,
    # and the target C subblock with its coefficients, adds 4 terms.
    
    # Actually, let me just directly enumerate all 28 terms for one 4x4x4 block.
    
    # Term structure: for each Strassen product (7 of them), 4 schoolbook terms for the 2x2 multiplication.
    
    # Let me define the Strassen products with their A-side and B-side linear combinations:
    # Each side is a sum of 0, 1, or 2 subblocks with coefficients +1 or -1.
    
    strassen_products = [
        # (name, A_side_blocks, B_side_blocks, C_side_blocks)
        # A_side_blocks: list of (subblock_name, coeff)
        # B_side_blocks: list of (subblock_name, coeff)
        # C_side_blocks: list of (subblock_name, coeff)
        ("M1", [("A11", 1), ("A22", 1)], [("B11", 1), ("B22", 1)], [("C11", 1), ("C22", 1)]),
        ("M2", [("A21", 1), ("A22", 1)], [("B11", 1)], [("C21", 1), ("C22", -1)]),
        ("M3", [("A11", 1)], [("B12", 1), ("B22", -1)], [("C12", 1), ("C22", 1)]),
        ("M4", [("A22", 1)], [("B21", 1), ("B11", -1)], [("C11", 1), ("C21", 1)]),
        ("M5", [("A11", 1), ("A12", 1)], [("B22", 1)], [("C11", -1), ("C12", 1)]),
        ("M6", [("A21", 1), ("A11", -1)], [("B11", 1), ("B12", 1)], [("C22", 1)]),
        ("M7", [("A12", 1), ("A22", -1)], [("B21", 1), ("B22", 1)], [("C11", 1)]),
    ]
    
    # Subblock row indices in U (for 4x4 A, row = i*4+j where i is row, j is col within 4x4)
    # But in the 4x4x4 block, A is 4x4, so U rows are i*4+j
    A_subblock_rows = {
        "A11": [0, 1, 4, 5],
        "A12": [2, 3, 6, 7],
        "A21": [8, 9, 12, 13],
        "A22": [10, 11, 14, 15],
    }
    
    # B subblock rows in V (for 4x4 B in the block, row = j*4+k)
    B_subblock_rows = {
        "B11": [0, 1, 8, 9],
        "B12": [2, 3, 10, 11],
        "B21": [8, 9, 12, 13],
        "B22": [10, 11, 14, 15],
    }
    
    # C subblock rows in W (for 4x4 C in the block, row = k*4+i)
    C_subblock_rows = {
        "C11": [0, 1, 4, 5],
        "C12": [8, 9, 12, 13],
        "C21": [2, 3, 6, 7],
        "C22": [10, 11, 14, 15],
    }
    
    # For each Strassen product, the A-side is a linear combination of subblocks.
    # For the 2x2 schoolbook within that product, each of the 4 (i',j',k') terms (where i',j',k' in 0,1)
    # contributes with the product of the A-side coefficient and B-side coefficient.
    
    # Let me build U_block, V_block, W_block for one 4x4x4 block.
    
    U_block = [[0.0] * 28 for _ in range(16)]
    V_block = [[0.0] * 28 for _ in range(16)]
    W_block = [[0.0] * 28 for _ in range(16)]
    
    col = 0
    for prod_name, A_blocks, B_blocks, C_blocks in strassen_products:
        # For this Strassen product, we have a 2x2x2 multiplication.
        # The A-side submatrix has entries: for each (i', j') in 0..1 x 0..1, 
        #   A_side[i',j'] = sum of coeffs * A_subblock[i' + 2*block_row, j' + 2*block_col]
        # where block_row, block_col depend on which subblock.
        
        # Actually, let me think of it differently. The 2x2 product (A_side)(B_side) = C_side
        # where A_side is a 2x2 matrix that is a linear combination of the 2x2 subblocks.
        
        # For the schoolbook decomposition of this 2x2x2 product:
        # For each (i', j', k') in {0,1}^3:
        #   U[i' + 2*block_a_row, col] += coeff_A
        #   V[j' + 2*block_b_row, col] += coeff_B  
        #   W[k' + 2*block_c_row, col] += 1  (but we need to handle the C-side coefficients)
        
        # Hmm, this is tricky because the C-side also has linear combinations.
        
        # Let me think about it more carefully. The Strassen identity says:
        # C_side = A_side * B_side
        # where each of A_side, B_side, C_side are 2x2 matrices that are linear combinations of subblocks.
        
        # For the tensor decomposition, I need:
        # sum_r U[row_a, r] * V[row_b, r] * W[row_c, r] = delta (appropriate)
        
        # For the schoolbook 2x2x2 within each Strassen term:
        # The contribution to C[i',k'] from A[i',j']*B[j',k'] is 1.
        
        # But A[i',j'] = sum_{block_a} coeff_a * A_subblock_a[i'+offset_a_row, j'+offset_a_col]
        # Similarly for B and C.
        
        # This means for each (i', j', k'), the actual U, V, W entries are:
        # U[A_subblock_a[i',j'], r] = coeff_a  for each block_a in A_blocks
        # V[B_subblock_b[j',k'], r] = coeff_b  for each block_b in B_blocks
        # W[C_subblock_c[k',i'], r] = coeff_c  for each block_c in C_blocks
        
        # But we need the product to work out correctly.
        
        # Let me just use a different approach: for each Strassen term, for each of the 4 schoolbook
        # (i',j',k') pairs, add one rank-1 term with the appropriate coefficients.
        
        for i_prime in range(2):
            for j_prime in range(2):
                for k_prime in range(2):
                    # U contribution: for each A subblock with coefficient
                    for a_name, a_coeff in A_blocks:
                        rows = A_subblock_rows[a_name]
                        # i' maps to row offset within the subblock
                        row_in_sub = i_prime * 2 + j_prime  # Wait, no.
                        # In a 2x2 subblock, element (i', j') is at position i'*2+j' in the flattened 4x4?
                        # No. The subblock is 2x2, and i', j' are indices within the subblock.
                        # In the 4x4 matrix, A11 occupies rows 0-1, cols 0-1.
                        # A11[i', j'] corresponds to A[i', j'] in the 4x4.
                        # So U row for A11[i',j'] is i'*4 + j' = i' + j' (since i',j' in 0,1, and 4x4 means row = i*4+j)
                        # Wait, U row index is i*m+j where m=4 for the 4x4 A.
                        # So A11[i',j'] -> U row i'*4+j'
                        # A12[i',j'] -> U row i'*4+(2+j')
                        # A21[i',j'] -> U row (2+i')*4+j'
                        # A22[i',j'] -> U row (2+i')*4+(2+j')
                        
                        # Let me recompute A_subblock_rows properly:
                        # A11: i in 0-1, j in 0-1 -> rows: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
                        # A12: i in 0-1, j in 2-3 -> rows: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
                        # A21: i in 2-3, j in 0-1 -> rows: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
                        # A22: i in 2-3, j in 2-3 -> rows: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
                        # This matches my earlier definition.
                        
                        # For a given (i', j'), the row in the subblock:
                        # A11: row = i'*4 + j'
                        # A12: row = i'*4 + 2 + j'
                        # A21: row = (2+i')*4 + j'
                        # A22: row = (2+i')*4 + 2 + j'
                        
                        # I'll compute the row directly.
                        if a_name == "A11":
                            u_row = i_prime * 4 + j_prime
                        elif a_name == "A12":
                            u_row = i_prime * 4 + 2 + j_prime
                        elif a_name == "A21":
                            u_row = (2 + i_prime) * 4 + j_prime
                        elif a_name == "A22":
                            u_row = (2 + i_prime) * 4 + 2 + j_prime
                        else:
                            raise ValueError(f"Unknown A block {a_name}")
                        
                        U_block[u_row][col] += a_coeff
                    
                    # V contribution
                    for b_name, b_coeff in B_blocks:
                        if b_name == "B11":
                            v_row = j_prime * 4 + k_prime
                        elif b_name == "B12":
                            v_row = j_prime * 4 + 2 + k_prime
                        elif b_name == "B21":
                            v_row = (2 + j_prime) * 4 + k_prime
                        elif b_name == "B22":
                            v_row = (2 + j_prime) * 4 + 2 + k_prime
                        else:
                            raise ValueError(f"Unknown B block {b_name}")
                        
                        V_block[v_row][col] += b_coeff
                    
                    # W contribution
                    for c_name, c_coeff in C_blocks:
                        if c_name == "C11":
                            w_row = k_prime * 4 + i_prime
                        elif c_name == "C12":
                            w_row = (2 + k_prime) * 4 + i_prime
                        elif c_name == "C21":
                            w_row = k_prime * 4 + 2 + i_prime
                        elif c_name == "C22":
                            w_row = (2 + k_prime) * 4 + 2 + i_prime
                        else:
                            raise ValueError(f"Unknown C block {c_name}")
                        
                        W_block[w_row][col] += c_coeff
                    
                    col += 1
    
    # Now col should be 28
    assert col == 28, f"Expected 28 columns, got {col}"
    
    # Now build the full 4x4x8 decomposition.
    # U: 16 rows, 56 columns
    # V: 32 rows, 56 columns  
    # W: 32 rows, 56 columns
    
    U = [[0.0] * 56 for _ in range(16)]
    V = [[0.0] * 56 for _ in range(32)]
    W = [[0.0] * 56 for _ in range(32)]
    
    # First block: B_left (cols 0-3 of B), C_left (cols 0-3 of C)
    for i in range(16):
        for c in range(28):
            U[i][c] = U_block[i][c]
            V[i][c] = V_block[i][c]
            W[i][c] = W_block[i][c]
    
    # Second block: B_right (cols 4-7 of B), C_right (cols 4-7 of C)
    # B_right[j, k] where j in 0-3, k in 4-7. In V, row = j*p + k = j*8 + k
    # For B_right, k = 4 + k' where k' in 0-3, so V row = j*8 + 4 + k'
    # C_right[i, k] where k in 4-7. In W, row = k*n + i = k*4 + i
    # For C_right, k = 4 + k' where k' in 0-3, so W row = (4+k')*4 + i
    
    for i in range(16):
        for c in range(28):
            U[i][28 + c] = U_block[i][c]
            # V: for B_right, the row offset is +16 compared to B_left
            # B_left[j,k] -> V row j*4+k; B_right[j,k'] -> V row j*8+(4+k')
            # j*8 + 4 + k' = j*4 + k' + 4*j + 4 ... hmm, let me just compute directly.
            # V_block row index r corresponds to B_left[j, k] where r = j*4+k
            # For B_right, it's V row j*8 + (4+k) = 8*j + 4 + k
            # So if V_block row is r = j*4+k, then B_right V row = 8*j + 4 + k
            # We need to map r -> 8*j + 4 + k where r = j*4 + k
            # j = r // 4, k = r % 4
            # B_right V row = 8*(r//4) + 4 + (r%4)
            
            for r in range(16):
                j_val = r // 4
                k_val = r % 4
                v_row_right = j_val * 8 + 4 + k_val
                V[v_row_right][28 + c] = V_block[r][c]
            
            # W: C_right[i, k] where k in 4-7. W row = k*4 + i
            # C_left[i, k'] where k' in 0-3. W row = k'*4 + i
            # W_block row r corresponds to C_left[i, k'] where r = k'*4 + i
            # For C_right, W row = (4+k')*4 + i = k'*4 + i + 16 = r + 16
            for r in range(16):
                W[r + 16][28 + c] = W_block[r][c]
    
    return {"U": U, "V": V, "W": W}
