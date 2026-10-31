import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen's algorithm for 4x4 matrix multiplication.
    # This decomposes the 4x4 multiplication into 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication has rank 7.
    # Total rank = 7 * 7 = 49.
    
    # The 2x2 Strassen decomposition (rank 7):
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map this to the global 4x4 indices.
    # A is 4x4, partitioned into 2x2 blocks A11, A12, A21, A22.
    # A11 is rows 0,1 cols 0,1.
    # A12 is rows 0,1 cols 2,3.
    # A21 is rows 2,3 cols 0,1.
    # A22 is rows 2,3 cols 2,3.
    
    # Similarly for B and C.
    
    # We will generate the U, V, W matrices for the 49 terms.
    # Each of the 7 "M" multiplications is itself a rank 7 decomposition.
    # So we have 7 * 7 = 49 columns in U, V, W.
    
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # Define the 7 Strassen terms for 2x2.
    # Each term is a set of coefficients for A, B, C.
    # For a 2x2 matrix, indices are 0..3.
    # A00, A01, A10, A11.
    
    # We will define the 7 linear forms for A, B, and the output C.
    # Then we combine them for the 4x4 case.
    
    # Strassen 2x2 coefficients (u, v, w) for each of the 7 multiplications.
    # u is 4x4 (for 2x2 A), v is 4x4 (for 2x2 B), w is 4x4 (for 2x2 C).
    # But we only need the non-zero parts.
    
    # Let's represent the 2x2 Strassen decomposition as a list of 7 tuples (u_2x2, v_2x2, w_2x2).
    # u_2x2 is a 4-dim vector (flat 2x2).
    
    # M1: (A00+A11)(B00+B11) -> C00, C01, C10, C11 contributions?
    # No, the multiplication M1 is a scalar.
    # The 2x2 multiplication C = AB is decomposed into 7 scalar multiplications M1..M7.
    # Then C is formed by linear combinations of M1..M7.
    
    # So for each of the 7 M's, we have:
    # M = (sum a_ij A_ij) * (sum b_ij B_ij)
    # And C_kl += coeff * M.
    
    # So the tensor term for M is:
    # u vector has a_ij.
    # v vector has b_ij.
    # w vector has coeff for C_kl.
    
    # Strassen 2x2:
    # M1: A: [1, 0, 0, 1], B: [1, 0, 0, 1], C: [1, 0, 0, 1] (for C00, C01, C10, C11? No, C is the result of M1? No.)
    # Wait, M1 is a scalar.
    # C00 = M1 + ...
    # So the term for M1 in the tensor is:
    # u: A coeffs for M1.
    # v: B coeffs for M1.
    # w: C coeffs for M1's contribution to C.
    # But M1 contributes to C00, C11?
    # C00 = M1 + M4 - M5 + M7
    # C11 = M1 - M2 + M3 + M6
    # So M1 contributes 1 to C00 and 1 to C11.
    # So w vector for M1 is [1, 0, 0, 1] (if C is flat [C00, C01, C10, C11]).
    
    # Let's define the 7 terms for 2x2.
    # Flat indices for 2x2: 00, 01, 10, 11 -> 0, 1, 2, 3.
    
    strassen_2x2 = [
        # M1
        (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
        # M2
        (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, 0])), # C10 += M2
        # M3
        (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 1, 0, 0])), # C01 += M3
        # M4
        (np.array([0, 0, 0, 1]), np.array([0, 0, 1, -1]), np.array([1, 0, 0, 0])), # C00 += M4
        # M5
        (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 0, 0, 0])), # C00 -= M5
        # M6
        (np.array([1, 0, -1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])), # C11 += M6
        # M7
        (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]))  # C00 += M7
    ]
    
    # Now, for 4x4, we partition into 2x2 blocks.
    # Block (bi, bj) of A corresponds to global indices.
    # bi=0, bj=0: A[0:2, 0:2] -> flat indices 0,1,4,5 (row-major 4x4)
    # bi=0, bj=1: A[0:2, 2:4] -> flat indices 2,3,6,7
    # bi=1, bj=0: A[2:4, 0:2] -> flat indices 8,9,12,13
    # bi=1, bj=1: A[2:4, 2:4] -> flat indices 10,11,14,15
    
    # We have 4 blocks of A, 4 blocks of B, 4 blocks of C.
    # The multiplication C = AB involves:
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    
    # Each of these 2x2 multiplications (e.g., A00 B00) is decomposed into 7 scalar multiplications.
    # So total 4 * 7 = 28? No.
    # Strassen for 4x4 uses 7 multiplications of 2x2 matrices.
    # The 2x2 matrices are linear combinations of the blocks.
    
    # M1_4x4 = (A00+A11)(B00+B11)
    # M2_4x4 = (A10+A11)B00
    # M3_4x4 = A00(B01-B11)
    # M4_4x4 = A11(B10-B00)
    # M5_4x4 = (A00+A01)B11
    # M6_4x4 = (A10-A00)(B00+B01)
    # M7_4x4 = (A01-A11)(B10+B11)
    
    # Each Mi_4x4 is a 2x2 matrix multiplication.
    # So each Mi_4x4 requires 7 scalar multiplications.
    # Total 7 * 7 = 49.
    
    # We will iterate over the 7 block-level Strassen terms.
    # For each, we define the linear forms for the 2x2 block inputs.
    # Then we expand each 2x2 block multiplication using the 2x2 Strassen decomposition.
    
    # Block indices:
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # For each M, we have a 2x2 multiplication X Y = Z.
    # X and Y are 2x2 matrices (linear combos of blocks).
    # Z is a 2x2 matrix (linear combo of C blocks).
    
    # We will use the 2x2 Strassen decomposition for Z = X Y.
    # This gives 7 scalar terms.
    
    # We need to map the 2x2 block indices to global 4x4 flat indices.
    
    def block_to_flat(block_idx, is_A):
        # block_idx is (bi, bj)
        bi, bj = block_idx
        # Global row start
        r0 = bi * 2
        c0 = bj * 2
        # Flat indices for the 2x2 block
        # Row 0: r0, r0+1; Col 0: c0, c0+1
        # Flat index = row * 4 + col
        return [
            r0 * 4 + c0,
            r0 * 4 + c0 + 1,
            (r0 + 1) * 4 + c0,
            (r0 + 1) * 4 + c0 + 1
        ]
    
    # Define the 7 block-level terms.
    # Each term is (X_blocks, Y_blocks, Z_blocks)
    # X_blocks is a dict mapping block_idx to coeff.
    
    block_terms = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}), # Z is C10
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1}), # Z is C01
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1}), # Z is C00
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1}), # Z is C00
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}), # Z is C11
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})  # Z is C00
    ]
    
    col = 0
    for X_blocks, Y_blocks, Z_blocks in block_terms:
        # For this block-level term, we have a 2x2 multiplication X Y = Z.
        # We decompose X Y = Z using 2x2 Strassen (7 terms).
        # For each of the 7 2x2 Strassen terms (u2, v2, w2):
        # u2 is coeffs for X (2x2).
        # v2 is coeffs for Y (2x2).
        # w2 is coeffs for Z (2x2).
        
        for u2, v2, w2 in strassen_2x2:
            # u2[k] is coeff for X element k.
            # X element k corresponds to a linear combo of A blocks.
            # X is formed by summing coeffs * A_blocks.
            # So the global A vector for this term is sum over blocks in X_blocks:
            # coeff_X_block * A_block_flat_indices * u2[k]
            
            # We build the global U, V, W vectors for this scalar term.
            U_vec = [0] * 16
            V_vec = [0] * 16
            W_vec = [0] * 16
            
            # X is a 2x2 matrix. Its elements are X00, X01, X10, X11.
            # u2 = [u2_00, u2_01, u2_10, u2_11]
            
            # X00 = sum_{block} coeff_X_block * A_block_00
            # X01 = sum_{block} coeff_X_block * A_block_01
            # etc.
            
            # So for each A block in X_blocks:
            for block_idx, coeff_X in X_blocks.items():
                flat_indices = block_to_flat(block_idx, True)
                # A_block_00 is at flat_indices[0]
                # A_block_01 is at flat_indices[1]
                # A_block_10 is at flat_indices[2]
                # A_block_11 is at flat_indices[3]
                
                U_vec[flat_indices[0]] += coeff_X * u2[0]
                U_vec[flat_indices[1]] += coeff_X * u2[1]
                U_vec[flat_indices[2]] += coeff_X * u2[2]
                U_vec[flat_indices[3]] += coeff_X * u2[3]
            
            # Similarly for Y and V
            for block_idx, coeff_Y in Y_blocks.items():
                flat_indices = block_to_flat(block_idx, False)
                V_vec[flat_indices[0]] += coeff_Y * v2[0]
                V_vec[flat_indices[1]] += coeff_Y * v2[1]
                V_vec[flat_indices[2]] += coeff_Y * v2[2]
                V_vec[flat_indices[3]] += coeff_Y * v2[3]
            
            # Similarly for Z and W
            for block_idx, coeff_Z in Z_blocks.items():
                flat_indices = block_to_flat(block_idx, False) # C blocks
                W_vec[flat_indices[0]] += coeff_Z * w2[0]
                W_vec[flat_indices[1]] += coeff_Z * w2[1]
                W_vec[flat_indices[2]] += coeff_Z * w2[2]
                W_vec[flat_indices[3]] += coeff_Z * w2[3]
            
            # Add to U, V, W matrices
            for i in range(16):
                U[i][col] = U_vec[i]
                V[i][col] = V_vec[i]
                W[i][col] = W_vec[i]
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
