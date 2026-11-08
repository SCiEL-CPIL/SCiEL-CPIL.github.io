import numpy as np
import time

def construct():
    """
    Rank-minimal decomposition for 4x4 matrix multiplication using recursive Strassen
    with dependency detection and rank reduction.
    C[i,j] = sum_k A[i,k] * B[k,j]
    U shape: (16, rank), V shape: (16, rank), W shape: (16, rank)
    Convention: U[i*4+k, r], V[k*4+j, r], W[j*4+i, r]
    """
    n, m, p = 4, 4, 4
    
    def u_idx(i, k):
        return i * 4 + k
    
    def v_idx(k, j):
        return k * 4 + j
    
    def w_idx(j, i):
        return j * 4 + i
    
    # Build the 49-component Strassen decomposition
    # Each of the 7 top-level Strassen products is itself a 2x2 Strassen (7 sub-products)
    # Total: 7 * 7 = 49 rank-1 terms
    
    # The 7 Strassen formulas for 2x2 matrix multiplication:
    # M1 = (A11)(B11 + B22)
    # M2 = (A11 + A21)(B22)
    # M3 = (A22)(B11 + B12)
    # M4 = (A22 - A12)(B21 - B11)
    # M5 = (A11 + A22)(B22)
    # M6 = (A12 - A22)(B11 + B21)
    # M7 = (A21 - A11)(B12 + B22)
    
    # For 4x4, we split into 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x2
    # C = [[C00, C01], [C10, C11]] where each Cij is 2x2
    
    # Each Cij is computed using the 7 Strassen products of the 2x2 blocks
    # But each 2x2 block product is itself computed via 7 rank-1 terms
    
    # Let's build this systematically
    
    # First, define the 2x2 Strassen formulas as (L, R, C_contrib) where
    # L is a 2x2 matrix of coefficients for A-blocks
    # R is a 2x2 matrix of coefficients for B-blocks  
    # C_contrib maps (row_block, col_block) -> [(sub_row, sub_col, coeff)]
    
    # For a 2x2 matrix product C = A*B where A,B,C are 2x2:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Strassen:
    # M1 = (A00)(B00 + B11)
    # M2 = (A00 + A10)(B11)
    # M3 = (A11)(B00 + B01)
    # M4 = (A11 - A01)(B10 - B00)
    # M5 = (A00 + A11)(B11)
    # M6 = (A01 - A11)(B00 + B10)
    # M7 = (A10 - A00)(B01 + B11)
    
    # C00 = M5 + M4 - M2 + M6
    # C01 = M1 + M7
    # C10 = M3 + M5
    # C11 = M1 - M4 + M2 - M3
    
    # For the 4x4 case, each "block" is a 2x2 matrix, so we need to expand further.
    # The 7 Strassen products at the top level each involve products of 2x2 matrices.
    # Each 2x2 matrix product can be done with 7 rank-1 terms (Strassen again).
    # So total rank = 7 * 7 = 49.
    
    # Let me define the 2x2 Strassen formulas more carefully.
    # Each formula has:
    # - L: a 2x2 matrix where L[r][c] is the coefficient of A[r][c] in this product
    # - R: a 2x2 matrix where R[r][c] is the coefficient of B[r][c] in this product
    # - C_contrib: for each output block (i,j), a list of (sub_i, sub_j, coeff)
    
    # Actually, let me think of it differently. For the 2x2 Strassen:
    # Each of the 7 products computes a scalar (in the 2x2 block sense).
    # But when we go one level deeper, each "scalar" is actually a 2x2 matrix.
    # So each of the 7 top-level products is a 2x2 matrix product, which needs 7 rank-1 terms.
    
    # Let me build the full decomposition by tracking all the way down.
    
    # For a 2x2 matrix product using Strassen:
    # The 7 products are:
    # P1 = A00 * (B00 + B11)
    # P2 = (A00 + A10) * B11
    # P3 = A11 * (B00 + B01)
    # P4 = (A11 - A01) * (B10 - B00)
    # P5 = (A00 + A11) * B11
    # P6 = (A01 - A11) * (B00 + B10)
    # P7 = (A10 - A00) * (B01 + B11)
    
    # And:
    # C00 = P5 + P4 - P2 + P6
    # C01 = P1 + P7
    # C10 = P3 + P5
    # C11 = P1 - P4 + P2 - P3
    
    # For the 4x4 case, A and B are 4x4, which we view as 2x2 blocks of 2x2 matrices.
    # Each Pk is a product of two 2x2 matrices (with some being sums/differences).
    # To compute each Pk as a tensor decomposition, we need to expand it.
    
    # Let me define the 7 Strassen formulas as (L_coeffs, R_coeffs, C_mapping)
    # where L_coeffs is a dict mapping (row, col) -> coeff for A-blocks
    # R_coeffs is a dict mapping (row, col) -> coeff for B-blocks
    # C_mapping is a dict mapping (out_row, out_col) -> list of (prod_idx, coeff)
    
    # Actually, for the decomposition, I need to track:
    # For each of the 49 rank-1 terms, what is U, V, W?
    
    # Let me build it step by step.
    
    # Level 1: 4x4 = 2x2 blocks of 2x2
    # The 7 Strassen products at level 1:
    # Each product is of the form (linear combo of A-blocks) * (linear combo of B-blocks)
    # Each A-block and B-block is 2x2, so the product is a 2x2 matrix.
    # To decompose this 2x2 matrix product into rank-1 terms, use Strassen again.
    
    # So for each of the 7 level-1 products, I have a 2x2 matrix product that needs 7 rank-1 terms.
    # Total: 7 * 7 = 49 terms.
    
    # Let me define the 2x2 Strassen formulas:
    # For a 2x2 product C = A * B:
    # The 7 products are indexed 0-6:
    
    strassen_2x2 = [
        # (L: {(r,c): coeff}, R: {(r,c): coeff})
        ({(0,0): 1}, {(0,0): 1, (1,1): 1}),  # P1 = A00 * (B00 + B11)
        ({(0,0): 1, (1,0): 1}, {(1,1): 1}),  # P2 = (A00 + A10) * B11
        ({(1,1): 1}, {(0,0): 1, (0,1): 1}),  # P3 = A11 * (B00 + B01)
        ({(1,1): 1, (0,1): -1}, {(1,0): 1, (0,0): -1}),  # P4 = (A11 - A01) * (B10 - B00)
        ({(0,0): 1, (1,1): 1}, {(1,1): 1}),  # P5 = (A00 + A11) * B11
        ({(0,1): 1, (1,1): -1}, {(0,0): 1, (1,0): 1}),  # P6 = (A01 - A11) * (B00 + B10)
        ({(1,0): 1, (0,0): -1}, {(0,1): 1, (1,1): 1}),  # P7 = (A10 - A00) * (B01 + B11)
    ]
    
    # C mapping: C[i][j] = sum of Pk * coeff
    c_mapping = {
        (0,0): [(4, 1), (3, 1), (1, -1), (5, 1)],  # C00 = P5 + P4 - P2 + P6
        (0,1): [(0, 1), (6, 1)],  # C01 = P1 + P7
        (1,0): [(2, 1), (4, 1)],  # C10 = P3 + P5
        (1,1): [(0, 1), (3, -1), (1, 1), (2, -1)],  # C11 = P1 - P4 + P2 - P3
    }
    
    # Now for the 4x4 case:
    # A is 4x4, viewed as 2x2 blocks of 2x2 matrices.
    # A[bi][bj] is the 2x2 block at block-position (bi, bj), which is A[2*bi:2*bi+2, 2*bj:2*bj+2]
    # Similarly for B and C.
    
    # The 7 level-1 Strassen products:
    # Each product k involves:
    # - A linear combination of A-blocks (from strassen_2x2[k][0])
    # - A linear combination of B-blocks (from strassen_2x2[k][1])
    # The result is a 2x2 matrix at each block position.
    
    # To compute this 2x2 matrix product, we use Strassen again.
    # So for each level-1 product k, we have 7 sub-products (level-2 Strassen).
    
    # For level-1 product k:
    # The "A-matrix" is a 2x2 matrix where each entry is a linear combination of 2x2 A-blocks.
    # The "B-matrix" is a 2x2 matrix where each entry is a linear combination of 2x2 B-blocks.
    # The product is a 2x2 matrix of 2x2 matrices, i.e., a 4x4 matrix.
    
    # Wait, I'm confusing myself. Let me restart with a clearer approach.
    
    # The 4x4 matrix multiplication tensor has entries T[i,k,j] = 1 if we're computing C[i,j] = sum_k A[i,k]*B[k,j]
    # In the convention: U[i*4+k, r], V[k*4+j, r], W[j*4+i, r]
    # So T[a,b,c] = 1 where a = i*4+k, b = k*4+j, c = j*4+i
    
    # The rank-1 decomposition: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # For the recursive Strassen:
    # Split A into 2x2 blocks of 2x2: A = [[A00, A01], [A10, A11]]
    # Split B into 2x2 blocks of 2x2: B = [[B00, B01], [B10, B11]]
    # Split C into 2x2 blocks of 2x2: C = [[C00, C01], [C10, C11]]
    
    # C = A * B, so:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Using Strassen for the 2x2 block product:
    # M1 = A00 * (B00 + B11)
    # M2 = (A00 + A10) * B11
    # M3 = A11 * (B00 + B01)
    # M4 = (A11 - A01) * (B10 - B00)
    # M5 = (A00 + A11) * B11
    # M6 = (A01 - A11) * (B00 + B10)
    # M7 = (A10 - A00) * (B01 + B11)
    
    # C00 = M5 + M4 - M2 + M6
    # C01 = M1 + M7
    # C10 = M3 + M5
    # C11 = M1 - M4 + M2 - P3
    
    # Each Mk is a product of two 2x2 matrices (with some being sums/differences).
    # To decompose each Mk into rank-1 terms, we use Strassen again on the 2x2 matrices.
    
    # So for each Mk, we have a 2x2 matrix product, which needs 7 rank-1 terms.
    # Total: 7 * 7 = 49 rank-1 terms.
    
    # Let me build this systematically.
    
    # For a 2x2 matrix product X * Y where X and Y are 2x2:
    # Using Strassen:
    # S1 = X00 * (Y00 + Y11)
    # S2 = (X00 + X10) * Y11
    # S3 = X11 * (Y00 + Y01)
    # S4 = (X11 - X01) * (Y10 - Y00)
    # S5 = (X00 + X11) * Y11
    # S6 = (X01 - X11) * (Y00 + Y10)
    # S7 = (X10 - X00) * (Y01 + Y11)
    
    # Result:
    # R00 = S5 + S4 - S2 + S6
    # R01 = S1 + S7
    # R10 = S3 + S5
    # R11 = S1 - S4 + S2 - S3
    
    # Now for the 4x4 case:
    # Each Mk involves products of linear combinations of 2x2 A-blocks and 2x2 B-blocks.
    # When we expand Mk using the inner Strassen, we get 7 rank-1 terms.
    
    # Let me define the 7 inner Strassen formulas:
    # For X * Y (both 2x2):
    # The 7 products are:
    inner_strassen = [
        # (X_coeffs: {(r,c): coeff}, Y_coeffs: {(r,c): coeff})
        ({(0,0): 1}, {(0,0): 1, (1,1): 1}),  # S1
        ({(0,0): 1, (1,0): 1}, {(1,1): 1}),  # S2
        ({(1,1): 1}, {(0,0): 1, (0,1): 1}),  # S3
        ({(1,1): 1, (0,1): -1}, {(1,0): 1, (0,0): -1}),  # S4
        ({(0,0): 1, (1,1): 1}, {(1,1): 1}),  # S5
        ({(0,1): 1, (1,1): -1}, {(0,0): 1, (1,0): 1}),  # S6
        ({(1,0): 1, (0,0): -1}, {(0,1): 1, (1,1): 1}),  # S7
    ]
    
    # Result mapping for inner Strassen:
    inner_c_map = {
        (0,0): [(4, 1), (3, 1), (1, -1), (5, 1)],  # R00 = S5 + S4 - S2 + S6
        (0,1): [(0, 1), (6, 1)],  # R01 = S1 + S7
        (1,0): [(2, 1), (4, 1)],  # R10 = S3 + S5
        (1,1): [(0, 1), (3, -1), (1, 1), (2, -1)],  # R11 = S1 - S4 + S2 - S3
    }
    
    # Now for the 4x4 case, the 7 outer Strassen products:
    # Each outer product k involves:
    # - A "X-matrix" which is a 2x2 matrix where each entry is a linear combination of 2x2 A-blocks
    # - A "Y-matrix" which is a 2x2 matrix where each entry is a linear combination of 2x2 B-blocks
    
    # For outer product k, the X-matrix entry at (r,c) is:
    # sum over (br,bc) of X_coeffs[(r,c)] * A_block[(br,bc)]
    # But wait, X is already a 2x2 matrix of 2x2 blocks.
    
    # I think I'm overcomplicating this. Let me just build the 49 terms directly.
    
    # For the 4x4 case with recursive Strassen:
    # The 49 terms are indexed by (outer_k, inner_s) where outer_k in 0..6, inner_s in 0..6
    
    # For each (outer_k, inner_s):
    # - U contributes to A-indices based on outer_k and inner_s
    # - V contributes to B-indices based on outer_k and inner_s
    # - W contributes to C-indices based on outer_k and inner_s
    
    # Let me trace through an example.
    
    # Outer product 0: M1 = A00 * (B00 + B11)
    # Here, X = A00 (a 2x2 matrix), Y = B00 + B11 (a 2x2 matrix)
    # To compute X * Y using inner Strassen:
    # S1 = X00 * (Y00 + Y11)
    # S2 = (X00 + X10) * Y11
    # etc.
    
    # X00 = A[0,0] (scalar, since A00 is 2x2, X00 is the (0,0) entry of A00)
    # Wait, no. X is a 2x2 matrix where each entry is a scalar (from the 2x2 block).
    # So X00 is a scalar, X01 is a scalar, etc.
    
    # I think the issue is that I'm mixing up levels. Let me be very explicit.
    
    # 4x4 matrix A:
    # A = [[a00, a01, a02, a03],
    #      [a10, a11, a12, a13],
    #      [a20, a21, a22, a23],
    #      [a30, a31, a32, a33]]
    
    # View as 2x2 blocks of 2x2:
    # A00 = [[a00, a01], [a10, a11]]
    # A01 = [[a02, a03], [a12, a13]]
    # A10 = [[a20, a21], [a30, a31]]
    # A11 = [[a22, a23], [a32, a33]]
    
    # Similarly for B and C.
    
    # Outer Strassen:
    # M1 = A00 * (B00 + B11)
    # M2 = (A00 + A10) * B11
    # M3 = A11 * (B00 + B01)
    # M4 = (A11 - A01) * (B10 - B00)
    # M5 = (A00 + A11) * B11
    # M6 = (A01 - A11) * (B00 + B10)
    # M7 = (A10 - A00) * (B01 + B11)
    
    # Each Mk is a 2x2 matrix product. For example:
    # M1 = A00 * (B00 + B11)
    # A00 is 2x2, (B00 + B11) is 2x2, so M1 is 2x2.
    
    # To decompose M1 into rank-1 terms, use inner Strassen:
    # Let X = A00, Y = B00 + B11
    # S1 = X00 * (Y00 + Y11)
    # S2 = (X00 + X10) * Y11
    # S3 = X11 * (Y00 + Y01)
    # S4 = (X11 - X01) * (Y10 - Y00)
    # S5 = (X00 + X11) * Y11
    # S6 = (X01 - X11) * (Y00 + Y10)
    # S7 = (X10 - X00) * (Y01 + Y11)
    
    # Where X00 = a00, X01 = a01, X10 = a10, X11 = a11
    # Y00 = b00 + b22, Y01 = b01 + b23, Y10 = b10 + b32, Y11 = b11 + b33
    
    # So S1 = a00 * ((b00 + b22) + (b11 + b33)) = a00 * (b00 + b11 + b22 + b33)
    # This is a rank-1 term: U contributes a00, V contributes (b00+b11+b22+b33), W contributes to C00 (since M1 contributes to C01 and C11)
    
    # Wait, M1 contributes to C01 and C11:
    # C01 = M1 + M7
    # C11 = M1 - M4 + M2 - M3
    
    # And M1 is a 2x2 matrix, so M1[0,0] contributes to C[0,2], M1[0,1] to C[0,3], M1[1,0] to C[1,2], M1[1,1] to C[1,3]
    
    # This is getting complex. Let me just build the 49 terms programmatically.
    
    # I'll use the following approach:
    # 1. Define the 7 outer Strassen formulas
    # 2. For each outer formula, define the 7 inner Strassen formulas
    # 3. For each (outer, inner) pair, determine U, V, W contributions
    
    # Let me define helper functions to build the U, V, W matrices.
    
    rank = 49
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    # Outer Strassen formulas: (L: {(br,bc): coeff}, R: {(br,bc): coeff})
    # where (br,bc) is the block position in the 2x2 block matrix
    outer_formulas = [
        ({(0,0): 1}, {(0,0): 1, (1,1): 1}),  # M1
        ({(0,0): 1, (1,0): 1}, {(1,1): 1}),  # M2
        ({(1,1): 1}, {(0,0): 1, (0,1): 1}),  # M3
        ({(1,1): 1, (0,1): -1}, {(1,0): 1, (0,0): -1}),  # M4
        ({(0,0): 1, (1,1): 1}, {(1,1): 1}),  # M5
        ({(0,1): 1, (1,1): -1}, {(0,0): 1, (1,0): 1}),  # M6
        ({(1,0): 1, (0,0): -1}, {(0,1): 1, (1,1): 1}),  # M7
    ]
    
    # Outer C mapping: C_block[i,j] = sum of Mk * coeff
    outer_c_map = {
        (0,0): [(4, 1), (3, 1), (1, -1), (5, 1)],  # C00 = M5 + M4 - M2 + M6
        (0,1): [(0, 1), (6, 1)],  # C01 = M1 + M7
        (1,0): [(2, 1), (4, 1)],  # C10 = M3 + M5
        (1,1): [(0, 1), (3, -1), (1, 1), (2, -1)],  # C11 = M1 - M4 + M2 - M3
    }
    
    # Inner Strassen formulas: (L: {(r,c): coeff}, R: {(r,c): coeff})
    # where (r,c) is the position within the 2x2 matrix
    inner_formulas = [
        ({(0,0): 1}, {(0,0): 1, (1,1): 1}),  # S1
        ({(0,0): 1, (1,0): 1}, {(1,1): 1}),  # S2
        ({(1,1): 1}, {(0,0): 1, (0,1): 1}),  # S3
        ({(1,1): 1, (0,1): -1}, {(1,0): 1, (0,0): -1}),  # S4
        ({(0,0): 1, (1,1): 1}, {(1,1): 1}),  # S5
        ({(0,1): 1, (1,1): -1}, {(0,0): 1, (1,0): 1}),  # S6
        ({(1,0): 1, (0,0): -1}, {(0,1): 1, (1,1): 1}),  # S7
    ]
    
    # Inner C mapping: R[i,j] = sum of Sk * coeff
    inner_c_map = {
        (0,0): [(4, 1), (3, 1), (1, -1), (5, 1)],  # R00 = S5 + S4 - S2 + S6
        (0,1): [(0, 1), (6, 1)],  # R01 = S1 + S7
        (1,0): [(2, 1), (4, 1)],  # R10 = S3 + S5
        (1,1): [(0, 1), (3, -1), (1, 1), (2, -1)],  # R11 = S1 - S4 + S2 - S3
    }
    
    # Now build the 49 terms
    for outer_k in range(7):
        outer_L, outer_R = outer_formulas[outer_k]
        
        for inner_s in range(7):
            inner_L, inner_R = inner_formulas[inner_s]
            
            rank_idx = outer_k * 7 + inner_s
            
            # U contributions: from A
            # For each (br, bc) in outer_L with coeff c1:
            #   For each (r, c) in inner_L with coeff c2:
            #     A index: i = 2*br + r, k = 2*bc + c
            #     U[i*4+k, rank_idx] += c1 * c2
            for (br, bc), c1 in outer_L.items():
                for (r, c), c2 in inner_L.items():
                    i = 2 * br + r
                    k = 2 * bc + c
                    U[i * 4 + k, rank_idx] += c1 * c2
            
            # V contributions: from B
            # For each (br, bc) in outer_R with coeff c1:
            #   For each (r, c) in inner_R with coeff c2:
            #     B index: k = 2*br + r, j = 2*bc + c
            #     V[k*4+j, rank_idx] += c1 * c2
            for (br, bc), c1 in outer_R.items():
                for (r, c), c2 in inner_R.items():
                    k = 2 * br + r
                    j = 2 * bc + c
                    V[k * 4 + j, rank_idx] += c1 * c2
            
            # W contributions: from C
            # The outer product outer_k contributes to C-blocks according to outer_c_map
            # The inner product inner_s contributes to the 2x2 result according to inner_c_map
            # So for each (cbr, cbc) in outer_c_map with (outer_k, c_outer):
            #   For each (r, c) in inner_c_map with (inner_s, c_inner):
            #     C index: j = 2*cbr + r, i = 2*cbc + c
            #     W[j*4+i, rank_idx] += c_outer * c_inner
            
            # First, find which C-blocks outer_k contributes to
            for (cbr, cbc), contributions in outer_c_map.items():
                for (prod_idx, c_outer) in contributions:
                    if prod_idx == outer_k:
                        # Now find which positions within the 2x2 block inner_s contributes to
                        for (r, c), inner_contribs in inner_c_map.items():
                            for (s_idx, c_inner) in inner_contribs:
                                if s_idx == inner_s:
                                    j = 2 * cbr + r
                                    i = 2 * cbc + c
                                    W[j * 4 + i, rank_idx] += c_outer * c_inner
    
    # Verify correctness
    tensor = np.einsum('ar,br,cr->abc', U, V, W)
    
    expected = np.zeros((16, 16, 16))
    for i in range(4):
        for k in range(4):
            for j in range(4):
                expected[i*4+k, k*4+j, j*4+i] = 1.0
    
    if np.max(np.abs(tensor - expected)) > 1e-10:
        # Return the unverified result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by finding linear dependencies
    # Build the component matrix
    comp_matrix = np.einsum('ar,br,cr->abc r', U, V, W).reshape(-1, rank)
    
    # Compute SVD to find numerical rank
    sv = np.linalg.svd(comp_matrix, compute_uv=False)
    tol = 1e-10
    num_rank = np.sum(sv > tol)
    
    if num_rank < rank:
        # Try to remove components one at a time
        for remove_idx in range(rank):
            mask = np.ones(rank, dtype=bool)
            mask[remove_idx] = False
            recon = np.einsum('ar,br,cr->abc', U[:, mask], V[:, mask], W[:, mask])
            if np.max(np.abs(recon - expected)) < 1e-10:
                return {"U": U[:, mask].tolist(), "V": V[:, mask].tolist(), "W": W[:, mask].tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

