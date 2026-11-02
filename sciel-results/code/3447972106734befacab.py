"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Block-Strassen decomposition for 6x6 matrices.
    # We treat the 6x6 matrix as a 2x2 block matrix of 3x3 blocks.
    # The rank of 3x3 matrix multiplication is 23 (Lader's algorithm).
    # The rank of 2x2 matrix multiplication is 7 (Strassen).
    # Total rank = 7 * 23 = 161? No, Strassen applies to the blocks.
    # Actually, we can use the 3x3 rank 23 decomposition inside the Strassen steps.
    # However, a simpler known construction for 6x6 is rank 189.
    # Let's use the standard Block-Strassen with 3x3 blocks.
    # Rank of 3x3 multiplication is 23.
    # Strassen's algorithm for 2x2 block matrices uses 7 multiplications.
    # Each multiplication is of 3x3 matrices, which takes rank 23.
    # Total rank = 7 * 23 = 161.
    # Wait, is 161 valid for 6x6?
    # Yes, if we can decompose 3x3 multiplication into rank 23.
    # Lader's algorithm gives rank 23 for 3x3.
    
    # Let's verify if we can construct the 3x3 rank 23 decomposition easily.
    # It is complex to hardcode Lader's 23-term decomposition.
    # Is there a simpler rank 189 decomposition?
    # Rank 189 comes from a different blocking or a known specific algorithm.
    # Actually, 6x6 rank 189 is a well-known bound.
    # Let's try to implement the 3x3 rank 23 (Lader) and Strassen.
    
    # Since hardcoding Lader's 23 terms is error-prone without the explicit coefficients,
    # and I don't have them memorized perfectly, I should check if a simpler 
    # construction exists or if I can use a known library/routine.
    # However, the prompt says "Python standard library, NumPy and SciPy are available".
    # It does not say I can use a tensor decomposition library.
    
    # Alternative: Use the 2x2 block Strassen with 3x3 blocks.
    # If I use schoolbook for 3x3 (rank 27), total rank = 7 * 27 = 189.
    # This is exactly rank 189!
    # And it is much easier to implement: just apply Strassen's formulas 
    # where each "multiplication" is a schoolbook 3x3 multiplication.
    
    # Let's implement Strassen for 2x2 blocks of size 3.
    
    # Strassen Formulas for C = A * B:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 3x3 matrix product.
    # We decompose each 3x3 product into rank 27 terms (schoolbook).
    # Total terms = 7 * 27 = 189.
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a schoolbook 3x3 multiplication block
    # offset: starting column index
    # a_block: 2x2 tuple of indices (row_start, col_start) for A in 6x6?
    # Actually, U, V, W are for the whole 6x6 tensor.
    # U index: i*m + j. V index: j*p + k. W index: k*n + i.
    # A is 6x6, B is 6x6.
    # A[i,j], B[j,k] -> C[i,k].
    
    # Let's define the blocks.
    # A11: rows 0-2, cols 0-2
    # A12: rows 0-2, cols 3-5
    # A21: rows 3-5, cols 0-2
    # A22: rows 3-5, cols 3-5
    # Same for B.
    
    def add_schoolbook_3x3(offset, a_r_start, a_c_start, b_r_start, b_c_start, c_r_start, c_c_start):
        """
        Adds 27 terms for C_sub = A_sub * B_sub.
        A_sub is 3x3 at (a_r_start, a_c_start) in A.
        B_sub is 3x3 at (b_r_start, b_c_start) in B.
        C_sub is 3x3 at (c_r_start, c_c_start) in C.
        """
        col = offset
        for i_local in range(3):
            for j_local in range(3):
                for k_local in range(3):
                    i = a_r_start + i_local
                    j = a_c_start + j_local
                    k = b_c_start + k_local # B col index is global k? 
                    # Wait, B is m x p. m=6, p=6.
                    # B[j, k].
                    # In the global tensor:
                    # U[i*m + j] = 1
                    # V[j*p + k] = 1
                    # W[k*n + i] = 1
                    # Note: W index is k*n + i.
                    
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return col
    
    # We need to handle sums and differences of blocks.
    # Strassen uses sums like A11 + A22.
    # This means U entries will be sums of contributions from different terms.
    # We can just accumulate the coefficients.
    
    # Let's define the 7 Strassen multiplications.
    # Each multiplication is of two 3x3 "virtual" matrices.
    # We need to map the virtual matrix entries to the global U, V, W.
    
    # Virtual Matrix A_M:
    # If A_M = A11 + A22, then A_M[i,j] = A[0+i, 0+j] + A[3+i, 3+j].
    # This corresponds to adding 1 to U[0+i, 0+j] and 1 to U[3+i, 3+j] for that term.
    
    # Let's create a helper to add a term for a virtual multiplication.
    # term_id: 0..6
    # a_def: list of (sign, r_start, c_start)
    # b_def: list of (sign, r_start, c_start)
    # c_def: list of (sign, r_start, c_start) -> This is where the result goes, with sign.
    # Actually, the result C is a sum of these virtual products.
    # C = S1*M1 + S2*M2 ...
    # So for each Strassen step, we have 3 matrices: A_v, B_v, C_v (contribution to C).
    # A_v is a sum of blocks. B_v is a sum of blocks.
    # The product A_v * B_v contributes to C with a coefficient.
    
    # Let's iterate through the 7 Strassen steps.
    
    # Step 1: M1 = (A11 + A22) * (B11 + B22)
    # A_v: [(1, 0, 0), (1, 3, 3)]
    # B_v: [(1, 0, 0), (1, 3, 3)]
    # C_v: [(1, 0, 0), (1, 3, 3)] # C11 and C22 get +M1
    
    # Step 2: M2 = (A21 + A22) * B11
    # A_v: [(1, 3, 0), (1, 3, 3)]
    # B_v: [(1, 0, 0)]
    # C_v: [(1, 3, 0), (1, 3, 3)] # C21 and C22 get +M2? 
    # C21 = M2 + M4. C22 = M1 - M2 + M3 + M6.
    # So C21 gets +1, C22 gets -1.
    
    # Step 3: M3 = A11 * (B12 - B22)
    # A_v: [(1, 0, 0)]
    # B_v: [(1, 0, 3), (-1, 3, 3)]
    # C_v: [(1, 0, 3), (1, 3, 3)] # C12 = M3 + M5. C22 = ... + M3 ...
    
    # Step 4: M4 = A22 * (B21 - B11)
    # A_v: [(1, 3, 3)]
    # B_v: [(1, 3, 0), (-1, 0, 0)]
    # C_v: [(1, 0, 0), (1, 3, 0)] # C11 = M1 + M4 ... C21 = M2 + M4
    
    # Step 5: M5 = (A11 + A12) * B22
    # A_v: [(1, 0, 0), (1, 0, 3)]
    # B_v: [(1, 3, 3)]
    # C_v: [(-1, 0, 0), (1, 0, 3)] # C11 = ... - M5 ... C12 = M3 + M5
    
    # Step 6: M6 = (A21 - A11) * (B11 + B12)
    # A_v: [(1, 3, 0), (-1, 0, 0)]
    # B_v: [(1, 0, 0), (1, 0, 3)]
    # C_v: [(1, 3, 3)] # C22 = ... + M6
    
    # Step 7: M7 = (A12 - A22) * (B21 + B22)
    # A_v: [(1, 0, 3), (-1, 3, 3)]
    # B_v: [(1, 3, 0), (1, 3, 3)]
    # C_v: [(1, 0, 0)] # C11 = M1 + M4 - M5 + M7
    
    steps = [
        # (A_def, B_def, C_def)
        # A_def: list of (coeff, r_start, c_start)
        # B_def: list of (coeff, r_start, c_start)
        # C_def: list of (coeff, r_start, c_start)
        ([ (1, 0, 0), (1, 3, 3) ], [ (1, 0, 0), (1, 3, 3) ], [ (1, 0, 0), (1, 3, 3) ]), # M1
        ([ (1, 3, 0), (1, 3, 3) ], [ (1, 0, 0) ], [ (1, 3, 0), (-1, 3, 3) ]), # M2
        ([ (1, 0, 0) ], [ (1, 0, 3), (-1, 3, 3) ], [ (1, 0, 3), (1, 3, 3) ]), # M3
        ([ (1, 3, 3) ], [ (1, 3, 0), (-1, 0, 0) ], [ (1, 0, 0), (1, 3, 0) ]), # M4
        ([ (1, 0, 0), (1, 0, 3) ], [ (1, 3, 3) ], [ (-1, 0, 0), (1, 0, 3) ]), # M5
        ([ (1, 3, 0), (-1, 0, 0) ], [ (1, 0, 0), (1, 0, 3) ], [ (1, 3, 3) ]), # M6
        ([ (1, 0, 3), (-1, 3, 3) ], [ (1, 3, 0), (1, 3, 3) ], [ (1, 0, 0) ])  # M7
    ]
    
    col_idx = 0
    for a_def, b_def, c_def in steps:
        # Each step is a product of virtual matrices.
        # The virtual matrices are 3x3.
        # We expand the product into 3*3*3 = 27 terms.
        # For each term (i,j,k) in the 3x3 product:
        # A_v[i,j] contributes to U[i_global, j_global]
        # B_v[j,k] contributes to V[j_global, k_global]
        # C_v[i,k] contributes to W[k_global, i_global]
        
        # We need to sum the coefficients for each global index.
        # Since A_v is a sum of blocks, A_v[i,j] is a sum of coeffs.
        # Same for B_v and C_v.
        
        # However, U, V, W are sparse. We can just add the contributions.
        # But wait, the term is a single column in U, V, W.
        # For a specific (i_local, j_local, k_local) in the 3x3 product:
        # The coefficient for U[i_global, j_global] is the sum of coeffs from a_def that cover (i_local, j_local).
        # But i_global and j_global are determined by the block.
        # Actually, for a fixed local i, j, k:
        # The contribution to the tensor is:
        # sum_{a in a_def} sum_{b in b_def} sum_{c in c_def} 
        #   coeff_a * coeff_b * coeff_c * 
        #   U[a.r + i, a.c + j] * V[b.r + j, b.c + k] * W[c.r + k, c.c + i]
        
        # This expands into many terms.
        # But we can just iterate through all combinations of blocks in a_def, b_def, c_def?
        # No, that would be incorrect. The product is linear.
        # (A1+A2)(B1+B2) = A1B1 + A1B2 + A2B1 + A2B2.
        # So we should expand the product first.
        
        # Let's expand the virtual product into a list of (coeff, a_block, b_block, c_block) terms.
        # Actually, it's simpler:
        # For each i_local in 0..2, j_local in 0..2, k_local in 0..2:
        #   We need to add a column to U, V, W.
        #   But the "value" at U[i_global, j_global] is not just 1, it's a sum.
        #   Wait, the decomposition is a sum of rank-1 tensors.
        #   Each rank-1 tensor has a single non-zero entry in U, V, W?
        #   No, a rank-1 tensor is u * v * w.
        #   If u has multiple non-zeros, it's still rank 1.
        #   But in the standard format, each column of U, V, W defines one rank-1 term.
        #   So for one column index `col`, U[:, col] is a vector, V[:, col] is a vector, W[:, col] is a vector.
        #   The term is sum_i sum_j sum_k U[i,j,col] V[j,k,col] W[k,i,col].
        
        # So for each local (i,j,k), we can define the vectors U_col, V_col, W_col.
        # U_col will have non-zeros at positions corresponding to the blocks in a_def.
        # V_col will have non-zeros at positions corresponding to the blocks in b_def.
        # W_col will have non-zeros at positions corresponding to the blocks in c_def.
        
        # But we must multiply the coefficients.
        # The coefficient for U[a.r+i, a.c+j] is coeff_a.
        # The coefficient for V[b.r+j, b.c+k] is coeff_b.
        # The coefficient for W[c.r+k, c.c+i] is coeff_c.
        
        # However, this single column represents the sum of all products of these blocks?
        # No.
        # If A_v = A1 + A2, and B_v = B1 + B2.
        # A_v * B_v = A1B1 + A1B2 + A2B1 + A2B2.
        # If we put A1 and A2 in the same U vector, and B1 and B2 in the same V vector,
        # the product (U dot V) will include cross terms like A1B2 and A2B1.
        # This is exactly what we want!
        # So we can just put the coefficients into the vectors.
        
        # For a fixed local (i, j, k):
        # U_col[0] = sum_{a in a_def, a covers (i,j)} coeff_a
        # V_col[0] = sum_{b in b_def, b covers (j,k)} coeff_b
        # W_col[0] = sum_{c in c_def, c covers (i,k)} coeff_c
        
        # Wait, the indices are global.
        # For a block a with start (ra, ca), the global index is (ra+i, ca+j).
        # If two blocks cover the same (i,j), their coefficients add up.
        # In our case, the blocks are disjoint in the 3x3 grid?
        # A11 is (0,0), A22 is (3,3). They are disjoint.
        # So for a given (i,j), only one block in a_def covers it?
        # Yes, because the blocks are 3x3 and placed at (0,0), (0,3), (3,0), (3,3).
        # So for a given (i,j) in 0..2, exactly one block covers it.
        
        # So:
        for i_local in range(3):
            for j_local in range(3):
                for k_local in range(3):
                    # Find coeff for U
                    u_coeff = 0
                    u_idx = -1
                    for coeff, r_s, c_s in a_def:
                        # Check if (i_local, j_local) is in this block
                        # The block is 3x3, so all local indices 0..2 are in the block.
                        # We just need to pick the right one.
                        # Since blocks are disjoint, we can just pick the first one?
                        # No, we need to know which global index to update.
                        # Actually, we can just accumulate.
                        # But since they are disjoint, only one will contribute?
                        # Wait, if A_v = A11 + A22, then for (0,0), A11 contributes. A22 does not.
                        # So we can just find the block that contains this local index.
                        # But all blocks contain all local indices 0..2.
                        # The difference is the global offset.
                        # So for (i_local, j_local), the global index for block (r_s, c_s) is (r_s+i_local, c_s+j_local).
                        # If we have multiple blocks, we have multiple global indices.
                        # We need to add to all of them?
                        # No, the vector U_col is a single vector.
                        # If A_v = A11 + A22, then U_col has non-zeros at both A11 positions and A22 positions.
                        # So we should add to all of them.
                        
                        # Let's re-evaluate.
                        # Term = U_col * V_col * W_col.
                        # U_col has 1 at A11[0,0] and 1 at A22[0,0].
                        # V_col has 1 at B11[0,0] and 1 at B22[0,0].
                        # Product includes A11[0,0]B11[0,0] and A11[0,0]B22[0,0] etc.
                        # This is correct for (A11+A22)(B11+B22).
                        
                        # So we just add the coefficient to the global index.
                        u_coeff += coeff
                        # But we need to store which global index.
                        # Since we are building the column, we can just update U directly.
                        U[r_s + i_local, c_s + j_local][col_idx] += coeff
                        # Wait, this adds to U. But we need to do this for all blocks in a_def.
                        # And we need to reset U for this column? No, we are building it.
                        # But we are inside the loop for (i,j,k).
                        # We should build the entire column for this (i,j,k).
                        # But the column is shared for all (i,j,k)?
                        # No, each (i,j,k) corresponds to a different term in the sum?
                        # No, the rank-1 term is defined by the vectors U_col, V_col, W_col.
                        # The vectors are fixed for the step, but they depend on (i,j,k)?
                        # No, the vectors are the same for all (i,j,k) if the blocks are the same?
                        # No, the global indices change with (i,j,k).
                        # So each (i,j,k) generates a new column in U, V, W.
                        # So col_idx increments for each (i,j,k).
                        
                    # We need to clear U, V, W for this column first?
                    # No, we are adding to it. But we should start fresh for each column.
                    # Let's reset the relevant parts? Or just use a dictionary?
                    # Since we are adding to specific indices, and the indices are unique for each (i,j,k) block combo?
                    # No, different (i,j,k) can hit the same global index if blocks overlap?
                    # They don't overlap.
                    # But for a fixed (i,j,k), we add to multiple global indices (one per block in a_def).
                    # So we should just add.
                    
                    # But we need to ensure we don't carry over from previous columns.
                    # We can just zero out the column before filling.
                    # Or, since we are using lists of lists, we can just set them.
                    # But we are accumulating.
                    # Let's just use a temporary dictionary for U, V, W for this column.
                    
                    u_dict = {}
                    v_dict = {}
                    w_dict = {}
                    
                    for coeff, r_s, c_s in a_def:
                        idx = (r_s + i_local) * m + (c_s + j_local)
                        u_dict[idx] = u_dict.get(idx, 0) + coeff
                        
                    for coeff, r_s, c_s in b_def:
                        idx = (r_s + j_local) * p + (c_s + k_local)
                        v_dict[idx] = v_dict.get(idx, 0) + coeff
                        
                    for coeff, r_s, c_s in c_def:
                        idx = (c_s + k_local) * n + (r_s + i_local)
                        w_dict[idx] = w_dict.get(idx, 0) + coeff
                        
                    # Now write to U, V, W
                    for idx, val in u_dict.items():
                        U[idx][col_idx] = val
                    for idx, val in v_dict.items():
                        V[idx][col_idx] = val
                    for idx, val in w_dict.items():
                        W[idx][col_idx] = val
                        
                    col_idx += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
