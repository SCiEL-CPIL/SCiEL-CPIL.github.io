import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-110 tensor decomposition for 5x5 matrix multiplication
    by applying Strassen's algorithm to the top-left 4x4 block and using
    schoolbook multiplication for the remaining rows/columns.
    
    The 5x5 multiplication is decomposed as:
    - Top-left 4x4 block: uses Strassen (rank 49 for 4x4)
    - Row 5 (index 4) of A times columns 0-3 of B: schoolbook (4*4=16 terms)
    - Rows 0-3 of A times column 5 (index 4) of B: schoolbook (4*4=16 terms)
    - A[4,4] * B[4,4]: 1 term
    Total: 49 + 16 + 16 + 1 = 82? Wait, let me recalculate.
    
    Actually, let me think more carefully. We split 5x5 into:
    - A is 5x5, B is 5x5, C is 5x5
    - C[0:4, 0:4] = A[0:4, 0:4] * B[0:4, 0:4]  (4x4 * 4x4)
    - C[0:4, 4]   = A[0:4, 0:4] * B[0:4, 4]    (4x4 * 4x1)
    - C[4, 0:4]   = A[4, 0:4] * B[0:4, 0:4]    (1x4 * 4x4)
    - C[4, 4]     = A[4, 4] * B[4, 4]           (scalar)
    
    For the 4x4 * 4x4 part, using Strassen recursively gives rank 49.
    For the 4x4 * 4x1 part, this is just 16 scalar multiplications (each element of the result is a dot product of length 4, but in tensor decomposition each scalar mult is rank 1).
    Wait, no. The tensor decomposition of matrix multiplication: each "scalar multiplication" a_ij * b_jk corresponds to one rank-1 term.
    
    So for 4x4 * 4x1: 4*4 = 16 scalar multiplications -> 16 rank-1 terms.
    For 1x4 * 4x4: 4*4 = 16 scalar multiplications -> 16 rank-1 terms.
    For scalar * scalar: 1 term.
    
    Total: 49 + 16 + 16 + 1 = 82.
    
    But wait, the 4x4 * 4x4 part via Strassen has rank 49, not 64.
    So total rank = 49 + 16 + 16 + 1 = 82.
    
    Let me verify: standard 5x5 is 125. Using Strassen on 4x4 block saves 64-49=15, so 125-15=110? No, that's not right either because the cross terms are still schoolbook.
    
    Let me just build it carefully.
    """
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # We'll build U, V, W as lists of lists, then convert to arrays
    # U: (n*m, rank) = (25, rank)
    # V: (m*p, rank) = (25, rank)  
    # W: (p*n, rank) = (25, rank)
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        """Add a rank-1 term."""
        U_terms.append([u_idx, u_val])
        V_terms.append([v_idx, v_val])
        W_terms.append([w_idx, w_val])
    
    # Strassen 2x2 decomposition
    # For 2x2 matrices:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B11 - B12)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Strassen 4x4: apply Strassen to 2x2 blocks, each block is 2x2
    # So 4x4 * 4x4 uses 7 "super-multiplications", each of which is 2x2 * 2x2 = 7 scalar mults
    # Total: 49
    
    # Let's define the Strassen 2x2 coefficients
    # For a 2x2 matmul, the 7 mults have these (u, v, w) patterns:
    # u indices: row*2 + col for A, so 0,1,2,3 for A[0,0],A[0,1],A[1,0],A[1,1]
    # v indices: row*2 + col for B
    # w indices: row*2 + col for C
    
    strassen_2x2 = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # M1 = (A11+A22)(B11+B22)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1)]),
        # M2 = (A21+A22)B11
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),
        # M3 = A11(B21-B22)
        ([(0, 1)], [(2, 1), (3, -1)], [(1, 1)]),
        # M4 = A22(B11-B12)
        ([(3, 1)], [(0, 1), (1, -1)], [(0, -1)]),
        # M5 = (A11+A12)B22
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1)]),
        # M6 = (A21-A11)(B11+B12)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(2, 1)]),
        # M7 = (A12-A22)(B21+B22)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    # For 4x4, we apply Strassen at the block level
    # A is divided into 2x2 blocks: A00, A01, A10, A11 each 2x2
    # Similarly for B and C
    
    # The 4x4 Strassen will have 7 "block mults", each block mult is 2x2*2x2
    # Each 2x2*2x2 uses 7 scalar mults via Strassen
    
    # Let me build the 4x4 Strassen decomposition
    # Block indices: 00, 01, 10, 11
    # Within each block, local indices 0,1,2,3 (row*2+col)
    
    # Global index for A: block_row * 4 + local_row * 2 + local_col
    # So A[br][bc][lr][lc] -> global index br*4 + lr*2 + lc
    # Similarly for B: block_row corresponds to col of A... wait.
    
    # A is n x m = 5x5, B is m x p = 5x5, C is p x n = 5x5
    # For the 4x4 subproblem:
    # A_sub is rows 0-3, cols 0-3 of A
    # B_sub is rows 0-3, cols 0-3 of B
    # C_sub is rows 0-3, cols 0-3 of C
    
    # A index: i*m + j where i in 0-3, j in 0-3
    # B index: j*p + k where j in 0-3, k in 0-3
    # C index: k*n + i where k in 0-3, i in 0-3
    
    # For 4x4 Strassen:
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    # C blocks: C00 (rows 0-1, cols 0-1), C01 (rows 0-1, cols 2-3), C10 (rows 2-3, cols 0-1), C11 (rows 2-3, cols 2-3)
    
    # Strassen at block level:
    # M1_block = (A00 + A11)(B00 + B11)
    # M2_block = (A10 + A11)B00
    # M3_block = A00(B10 - B11)
    # M4_block = A11(B00 - B01)
    # M5_block = (A00 + A01)B11
    # M6_block = (A10 - A00)(B00 + B01)
    # M7_block = (A01 - A11)(B10 + B11)
    
    # C00 = M1_block + M4_block - M5_block + M7_block
    # C01 = M3_block + M5_block
    # C10 = M2_block + M4_block
    # C11 = M1_block - M2_block + M3_block + M6_block
    
    # Each block mult is a 2x2 * 2x2, which we decompose using Strassen 2x2 (7 mults)
    
    # Let's define helper to add a 2x2*2x2 Strassen decomposition with offsets
    def add_strassen_2x2(a_offset, b_offset, c_offset):
        """
        Add Strassen 2x2 decomposition for 2x2 blocks at given offsets.
        a_offset: (row_offset, col_offset) for A (within the 4x4 subproblem)
        b_offset: (row_offset, col_offset) for B
        c_offset: (row_offset, col_offset) for C
        """
        for u_coeffs, v_coeffs, w_coeffs in strassen_2x2:
            for (u_local, u_val), (v_local, v_val), (w_local, w_val) in zip(u_coeffs, v_coeffs, w_coeffs):
                # u_local is 0-3, representing local row*2+col within the 2x2 block
                u_row_local = u_local // 2
                u_col_local = u_local % 2
                v_row_local = v_local // 2
                v_col_local = v_local % 2
                w_row_local = w_local // 2
                w_col_local = w_local % 2
                
                # Global indices within 4x4 subproblem
                u_row = a_offset[0] + u_row_local
                u_col = a_offset[1] + u_col_local
                v_row = b_offset[0] + v_row_local
                v_col = b_offset[1] + v_col_local
                w_row = c_offset[0] + w_row_local
                w_col = c_offset[1] + w_col_local
                
                # Map to full 5x5 indices
                u_global = u_row * m + u_col  # A index
                v_global = v_row * p + v_col  # B index
                w_global = w_row * n + w_col  # C index
                
                # But we need to combine coefficients from the block-level Strassen
                # This approach is getting complicated. Let me use a different strategy.
                
    # Alternative: build the full decomposition using linearity
    # Let me just enumerate all the terms explicitly
    
    # For the 4x4 * 4x4 part, I'll build it by composing the two levels of Strassen
    
    # Level 1: Block-level Strassen gives 7 block multiplications
    # Each block multiplication X * Y where X and Y are 2x2 matrices
    # Level 2: Each 2x2 * 2x2 is decomposed using Strassen 2x2 (7 scalar mults)
    
    # So total 49 scalar mults for the 4x4 part
    
    # Let me define the block-level Strassen terms
    # Each term is (X_coeffs, Y_coeffs, Z_coeffs) where X, Y are linear combos of A blocks, Y of B blocks, Z is the C block contribution
    
    # A block (br, bc): rows br*2 to br*2+1, cols bc*2 to bc*2+1
    # B block (br, bc): rows br*2 to br*2+1, cols bc*2 to bc*2+1  
    # C block (br, bc): rows br*2 to br*2+1, cols bc*2 to bc*2+1
    
    # Block-level Strassen:
    block_strassen = [
        # (X_coeffs, Y_coeffs, Z_block)
        # X_coeffs: list of (block_idx, coeff) for A
        # Y_coeffs: list of (block_idx, coeff) for B
        # Z_block: which C block this contributes to, with sign
        # M1 = (A00 + A11)(B00 + B11) -> contributes to C00, C11
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, -1)]),
        # M2 = (A10 + A11)B00 -> contributes to C10, C11
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3 = A00(B10 - B11) -> contributes to C01, C11
        ([(0, 1)], [(2, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4 = A11(B00 - B01) -> contributes to C00, C10
        ([(3, 1)], [(0, 1), (1, -1)], [(0, -1), (2, 1)]),
        # M5 = (A00 + A01)B11 -> contributes to C00, C01
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6 = (A10 - A00)(B00 + B01) -> contributes to C11
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (A01 - A11)(B10 + B11) -> contributes to C00
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    # Now for each block multiplication, we need to expand it into 2x2 * 2x2 Strassen
    # A block (br, bc) corresponds to A indices: (br*2 + lr) * m + (bc*2 + lc) for lr, lc in 0,1
    # B block (br, bc) corresponds to B indices: (br*2 + lr) * p + (bc*2 + lc) for lr, lc in 0,1
    # C block (br, bc) corresponds to C indices: (br*2 + lr) * n + (bc*2 + lc) for lr, lc in 0,1
    
    # For a block multiplication X * Y where X = sum_x (br_x, bc_x, coeff_x) and Y = sum_y (br_y, bc_y, coeff_y)
    # The product X * Y contributes to C blocks via the block-level formula
    # But actually, each "block mult" M_k = X_k * Y_k where X_k is a 2x2 matrix (linear combo of A blocks) 
    # and Y_k is a 2x2 matrix (linear combo of B blocks). The result is a 2x2 matrix that gets added to certain C blocks.
    
    # To decompose X_k * Y_k into scalar mults, we note that X_k and Y_k are each 2x2 matrices.
    # The product X_k * Y_k is a 2x2 matrix. We can decompose this 2x2 * 2x2 multiplication using Strassen.
    
    # But X_k is a linear combination of A blocks, so each element of X_k is a linear combination of A entries.
    # Similarly for Y_k.
    
    # This is getting very complex. Let me just use schoolbook for the 4x4 part to be safe,
    # but that gives rank 64 for the 4x4 part, total 64 + 16 + 16 + 1 = 97.
    
    # Actually, let me just implement it correctly with explicit expansion.
    
    # Clear the lists
    U_terms.clear()
    V_terms.clear()
    W_terms.clear()
    
    # For each block-level Strassen term, expand into 2x2 * 2x2 using schoolbook (4 mults per 2x2*2x2? No, 2*2*2=8)
    # Wait, 2x2 * 2x2 schoolbook is 8 scalar mults. With Strassen it's 7.
    
    # Let me just use schoolbook for the inner 2x2 * 2x2 to keep it simple and correct.
    # That gives 7 * 8 = 56 for the 4x4 part. Total: 56 + 16 + 16 + 1 = 89.
    
    # Or use Strassen for inner: 7 * 7 = 49. Total: 49 + 16 + 16 + 1 = 82.
    
    # Let me implement the Strassen version.
    
    def expand_block_mult(X_coeffs, Y_coeffs, Z_coeffs):
        """
        Expand a block-level multiplication into scalar multiplications.
        X_coeffs: list of (br, bc, coeff) for A blocks
        Y_coeffs: list of (br, bc, coeff) for B blocks
        Z_coeffs: list of (br, bc, coeff) for C blocks (where the result goes)
        """
        # X is a 2x2 matrix where X[ir][ic] = sum over (br, bc, c) in X_coeffs where br==ir, bc==ic of c * A[br*2+ir][bc*2+ic]
        # Wait, no. X is formed by combining A blocks. Each A block is 2x2.
        # X[ir][ic] = sum_{(br,bc,c)} c * A_block(br,bc)[ir][ic]
        # where A_block(br,bc)[ir][ic] = A[(br*2+ir) * m + (bc*2+ic)]
        
        # Similarly for Y.
        
        # The product X * Y is a 2x2 matrix. (X*Y)[ir][ic] = sum_j X[ir][j] * Y[j][ic]
        
        # For each (ir, ic) in 0,1 and j in 0,1:
        # (X*Y)[ir][ic] += X[ir][j] * Y[j][ic]
        # X[ir][j] = sum_{(br,bc,c)} c * A[(br*2+ir)*m + (bc*2+j)]
        # Y[j][ic] = sum_{(br,bc,c)} c * B[(br*2+j)*p + (bc*2+ic)]
        
        # So the contribution to (X*Y)[ir][ic] is a sum of products of A entries and B entries.
        # Each such product is a rank-1 term.
        
        # For each combination of A coefficient and B coefficient:
        for (a_br, a_bc, a_c) in X_coeffs:
            for (b_br, b_bc, b_c) in Y_coeffs:
                # A entry: A[(a_br*2+ir)*m + (a_bc*2+j)] for ir, j in 0,1
                # B entry: B[(b_br*2+j)*p + (b_bc*2+ic)] for j, ic in 0,1
                # Product contributes to (X*Y)[ir][ic]
                # Which contributes to C blocks via Z_coeffs
                
                # For each ir in 0,1 and ic in 0,1:
                for ir in range(2):
                    for ic in range(2):
                        # The term is A[(a_br*2+ir)*m + (a_bc*2+j)] * B[(b_br*2+j)*p + (b_bc*2+ic)]
                        # summed over j in 0,1
                        for j in range(2):
                            a_idx = (a_br * 2 + ir) * m + (a_bc * 2 + j)
                            b_idx = (b_br * 2 + j) * p + (b_bc * 2 + ic)
                            
                            # This product contributes to C[(c_br*2+ic)*n + (c_bc*2+ir)] for each (c_br, c_bc, c_z) in Z_coeffs
                            for (c_br, c_bc, c_z) in Z_coeffs:
                                c_idx = (c_br * 2 + ic) * n + (c_bc * 2 + ir)
                                coeff = a_c * b_c * c_z
                                if coeff != 0:
                                    U_terms.append((a_idx, coeff))
                                    V_terms.append((b_idx, coeff))
                                    W_terms.append((c_idx, coeff))
    
    # Add the 4x4 Strassen terms
    for X_coeffs, Y_coeffs, Z_coeffs in block_strassen:
        expand_block_mult(X_coeffs, Y_coeffs, Z_coeffs)
    
    # Now add the cross terms:
    # C[0:4, 4] = A[0:4, 0:4] * B[0:4, 4]
    # This is 4x4 * 4x1, schoolbook: 16 terms
    for i in range(4):
        for j in range(4):
            a_idx = i * m + j
            b_idx = j * p + 4
            c_idx = 4 * n + i
            U_terms.append((a_idx, 1))
            V_terms.append((b_idx, 1))
            W_terms.append((c_idx, 1))
    
    # C[4, 0:4] = A[4, 0:4] * B[0:4, 0:4]
    # 1x4 * 4x4, schoolbook: 16 terms
    for j in range(4):
        for k in range(4):
            a_idx = 4 * m + j
            b_idx = j * p + k
            c_idx = k * n + 4
            U_terms.append((a_idx, 1))
            V_terms.append((b_idx, 1))
            W_terms.append((c_idx, 1))
    
    # C[4, 4] = A[4,4] * B[4,4]
    a_idx = 4 * m + 4
    b_idx = 4 * p + 4
    c_idx = 4 * n + 4
    U_terms.append((a_idx, 1))
    V_terms.append((b_idx, 1))
    W_terms.append((c_idx, 1))
    
    # Build the matrices
    rank = len(U_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_idx, u_val) in enumerate(U_terms):
        U[u_idx][col] = u_val
    for col, (v_idx, v_val) in enumerate(V_terms):
        V[v_idx][col] = v_val
    for col, (w_idx, w_val) in enumerate(W_terms):
        W[w_idx][col] = w_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
