"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _strassen_4x4():
    """
    Return a rank-49 decomposition for 4x4 matrix multiplication using
    a two-level Strassen construction.
    
    The matrix multiplication tensor for 4x4 matrices can be decomposed into
    49 rank-1 tensors. This is achieved by viewing the 4x4 matrices as 2x2 
    blocks of 2x2 matrices, applying Strassen's 7-product algorithm to the 
    blocks, and then applying Strassen's 7-product algorithm again to each 
    of the resulting 2x2 matrix multiplications.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 formulas for 2x2 matmul:
    # Given X = [[x0, x1], [x2, x3]], Y = [[y0, y1], [y2, y3]]
    # s1 = (x0+x3)*(y0+y3)
    # s2 = (x2+x3)*y0
    # s3 = x0*(y1-y3)
    # s4 = x3*(y2-y0)
    # s5 = (x0+x1)*y3
    # s6 = (x2-x0)*(y0+y1)
    # s7 = (x1-x3)*(y2+y3)
    # Result:
    # [[s1+s4-s5+s7, s3+s5],
    #  [s2+s4, s1-s2+s3+s6]]
    
    # We define the 7 block products M1..M7 for the 4x4 case.
    # Each Mj is a 2x2 matrix product Xj * Yj.
    # Xj and Yj are 2x2 matrices whose entries are linear combinations of 2x2 sub-blocks of A and B.
    
    # Let A = [[A00, A01], [A10, A11]] and B = [[B00, B01], [B10, B11]]
    # where Aij, Bij are 2x2 matrices.
    
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B01-B11)
    # M4 = A11 * (B10-B00)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B01)
    # M7 = (A01-A11) * (B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Mj, we need to decompose the 2x2 matrix product into 7 scalar products.
    # We will iterate over the 7 block products (j=0..6) and for each, 
    # iterate over the 7 scalar Strassen steps (s=0..6).
    
    # Helper to get the 2x2 sub-block of A given block row br and block col bc
    # A is stored as a flat list of 16 elements: A[i*4+j]
    # Sub-block (br, bc) consists of A[br*2 + r, bc*2 + c] for r,c in 0..1
    # The indices in U are i*m + j = i*4 + j.
    
    def get_A_indices(br, bc):
        """Returns list of (flat_index, coeff) for the 2x2 block A[br, bc]"""
        # Block A[br, bc] has rows 2*br, 2*br+1 and cols 2*bc, 2*bc+1
        indices = []
        for r in range(2):
            for c in range(2):
                i = 2 * br + r
                j = 2 * bc + c
                indices.append((i * 4 + j, 1))
        return indices

    def get_B_indices(br, bc):
        """Returns list of (flat_index, coeff) for the 2x2 block B[br, bc]"""
        # Block B[br, bc] has rows 2*br, 2*br+1 and cols 2*bc, 2*bc+1
        indices = []
        for r in range(2):
            for c in range(2):
                j = 2 * br + r
                k = 2 * bc + c
                indices.append((j * 4 + k, 1))
        return indices

    # Define the 7 block products.
    # Each product is defined by (X_factors, Y_factors)
    # X_factors is a list of 4 items (x0, x1, x2, x3) for the 2x2 matrix X
    # Each item is a list of (block_index, coeff) where block_index is (br, bc)
    # We need to map these to flat indices.
    
    # Let's represent a 2x2 matrix of blocks as a 2x2 array of lists of (block_coords, coeff)
    
    def combine_blocks(blocks):
        """
        blocks: list of ((br, bc), coeff)
        Returns a flat list of (flat_index, coeff) for A
        """
        result = []
        for (br, bc), coeff in blocks:
            for idx, c in get_A_indices(br, bc):
                result.append((idx, c * coeff))
        return result

    def combine_blocks_B(blocks):
        """
        blocks: list of ((br, bc), coeff)
        Returns a flat list of (flat_index, coeff) for B
        """
        result = []
        for (br, bc), coeff in blocks:
            for idx, c in get_B_indices(br, bc):
                result.append((idx, c * coeff))
        return result

    # Define the 7 Strassen steps for the block level.
    # X is 2x2 blocks: X00, X01, X10, X11
    # Y is 2x2 blocks: Y00, Y01, Y10, Y11
    
    block_products = []
    
    # M1: X = [[A00, 0], [0, A11]] + ... wait, Strassen uses linear combinations.
    # M1 = (A00 + A11) * (B00 + B11)
    # X1 = [[A00+A11, 0], [0, 0]]? No.
    # The "matrix" X for M1 is not a 2x2 matrix of blocks in the standard Strassen sense.
    # Strassen computes 7 products of 2x2 matrices.
    # Here, the "2x2 matrices" are the blocks themselves.
    # So M1 is the product of two 2x2 matrices X1 and Y1.
    # X1 = [[A00+A11, 0], [0, 0]]? No, that's not right.
    # Strassen's 7 products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Mj is a 2x2 matrix product. Let's call the left factor Xj and right factor Yj.
    # Xj and Yj are 2x2 matrices whose entries are 2x2 blocks (or sums/diffs of them).
    # Wait, the entries of Xj and Yj are NOT 2x2 blocks. They are 2x2 matrices.
    # The product Xj * Yj is a 2x2 matrix (which is a block of C).
    # So Xj and Yj are 2x2 matrices.
    # But Xj and Yj are formed from the blocks of A and B.
    # For M1, X1 = [[A00+A11, 0], [0, 0]]? No.
    # The formula is (A00+A11) * (B00+B11).
    # This is the product of two 2x2 matrices.
    # Let X1 = [[A00+A11, 0], [0, 0]]? No, that would be a 4x4 matrix.
    # A00 is 2x2. A00+A11 is 2x2.
    # So X1 is a 2x2 matrix? No, X1 is a 2x2 matrix whose entries are scalars?
    # No, in the context of block Strassen, the "elements" of the block matrix are 2x2 matrices.
    # So the "scalar" multiplication in the Strassen formula is actually a 2x2 matrix multiplication.
    # So M1 is the product of two 2x2 matrices: L1 = A00+A11 and R1 = B00+B11.
    # But L1 and R1 are 2x2 matrices. Their product is a 2x2 matrix.
    # So M1 is a 2x2 matrix.
    # This matches the block structure of C.
    
    # So, for each of the 7 block products Mj, we have two 2x2 matrices Lj and Rj.
    # We then decompose the 2x2 matrix product Lj * Rj into 7 scalar products using Strassen.
    
    # Let's define Lj and Rj for j=1..7.
    # L1 = A00 + A11
    # R1 = B00 + B11
    
    # L2 = A10 + A11
    # R2 = B00
    
    # L3 = A00
    # R3 = B01 - B11
    
    # L4 = A11
    # R4 = B10 - B00
    
    # L5 = A00 + A01
    # R5 = B11
    
    # L6 = A10 - A00
    # R6 = B00 + B01
    
    # L7 = A01 - A11
    # R7 = B10 + B11
    
    # For each Lj, Rj, we apply the 7 scalar Strassen formulas.
    # Let Lj = [[l00, l01], [l10, l11]] and Rj = [[r00, r01], [r10, r11]].
    # The 7 scalar products are:
    # s1 = (l00+l11)(r00+r11)
    # s2 = (l10+l11)r00
    # s3 = l00(r01-r11)
    # s4 = l11(r10-r00)
    # s5 = (l00+l01)r11
    # s6 = (l10-l00)(r00+r01)
    # s7 = (l01-l11)(r10+r11)
    
    # The result Mj = Lj * Rj has entries:
    # Mj00 = s1 + s4 - s5 + s7
    # Mj01 = s3 + s5
    # Mj10 = s2 + s4
    # Mj11 = s1 - s2 + s3 + s6
    
    # Now we need to map the scalar entries l00, l01, etc., to the flat indices of A.
    # l00 is the (0,0) entry of Lj.
    # Lj is formed from blocks.
    # For example, L1 = A00 + A11.
    # A00 is the block at (0,0). Its (0,0) entry is A[0,0].
    # A11 is the block at (1,1). Its (0,0) entry is A[2,2].
    # So l00 for L1 is A[0,0] + A[2,2].
    
    # We will create a function to get the linear combination of A entries for a given
    # block expression and position (r, c) within the 2x2 block.
    
    def get_A_entry_linear_comb(block_expr, r, c):
        """
        block_expr: list of ((br, bc), coeff)
        r, c: position within the 2x2 block
        Returns list of (flat_index, coeff)
        """
        result = []
        for (br, bc), coeff in block_expr:
            i = 2 * br + r
            j = 2 * bc + c
            result.append((i * 4 + j, coeff))
        return result

    def get_B_entry_linear_comb(block_expr, r, c):
        """
        block_expr: list of ((br, bc), coeff)
        r, c: position within the 2x2 block
        Returns list of (flat_index, coeff)
        """
        result = []
        for (br, bc), coeff in block_expr:
            j = 2 * br + r
            k = 2 * bc + c
            result.append((j * 4 + k, coeff))
        return result

    # Define the block expressions for Lj and Rj
    # Each is a list of ((br, bc), coeff)
    
    L_exprs = [
        [((0,0), 1), ((1,1), 1)],  # L1
        [((1,0), 1), ((1,1), 1)],  # L2
        [((0,0), 1)],              # L3
        [((1,1), 1)],              # L4
        [((0,0), 1), ((0,1), 1)],  # L5
        [((1,0), 1), ((0,0), -1)], # L6
        [((0,1), 1), ((1,1), -1)]  # L7
    ]
    
    R_exprs = [
        [((0,0), 1), ((1,1), 1)],  # R1
        [((0,0), 1)],              # R2
        [((0,1), 1), ((1,1), -1)], # R3
        [((1,0), 1), ((0,0), -1)], # R4
        [((1,1), 1)],              # R5
        [((0,0), 1), ((0,1), 1)],  # R6
        [((1,0), 1), ((1,1), 1)]   # R7
    ]
    
    # Strassen coefficients for the scalar level
    # s1: l00+l11, r00+r11
    # s2: l10+l11, r00
    # s3: l00, r01-r11
    # s4: l11, r10-r00
    # s5: l00+l01, r11
    # s6: l10-l00, r00+r01
    # s7: l01-l11, r10+r11
    
    # We define the 7 scalar steps as (L_positions, R_positions)
    # L_positions: list of (r, c, coeff)
    # R_positions: list of (r, c, coeff)
    
    scalar_steps = [
        ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]), # s1
        ([ (1,0,1), (1,1,1) ], [ (0,0,1) ]),           # s2
        ([ (0,0,1) ], [ (0,1,1), (1,1,-1) ]),          # s3
        ([ (1,1,1) ], [ (1,0,1), (0,0,-1) ]),          # s4
        ([ (0,0,1), (0,1,1) ], [ (1,1,1) ]),           # s5
        ([ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ]), # s6
        ([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ])  # s7
    ]
    
    # Result coefficients for Mj entries
    # Mj00: s1(+1), s4(+1), s5(-1), s7(+1)
    # Mj01: s3(+1), s5(+1)
    # Mj10: s2(+1), s4(+1)
    # Mj11: s1(+1), s2(-1), s3(+1), s6(+1)
    
    result_coeffs = [
        { (0,0): 1, (1,1): 1 },    # s1
        { (1,0): 1, (1,1): -1 },   # s2
        { (0,1): 1, (1,1): 1 },    # s3
        { (0,0): 1, (1,0): 1 },    # s4
        { (0,0): -1, (0,1): 1 },   # s5
        { (1,1): 1 },              # s6
        { (0,0): 1 }               # s7
    ]
    
    # Block result coefficients
    # C00: M1(+1), M4(+1), M5(-1), M7(+1)
    # C01: M3(+1), M5(+1)
    # C10: M2(+1), M4(+1)
    # C11: M1(+1), M2(-1), M3(+1), M6(+1)
    
    block_result_coeffs = [
        { (0,0): 1, (1,1): 1 },    # M1
        { (1,0): 1, (1,1): -1 },   # M2
        { (0,1): 1, (1,1): 1 },    # M3
        { (0,0): 1, (1,0): 1 },    # M4
        { (0,0): -1, (0,1): 1 },   # M5
        { (1,1): 1 },              # M6
        { (0,0): 1 }               # M7
    ]
    
    col = 0
    for j in range(7):
        L_expr = L_exprs[j]
        R_expr = R_exprs[j]
        
        for s in range(7):
            L_pos, R_pos = scalar_steps[s]
            
            # Build U vector
            u_dict = {}
            for r, c, coeff in L_pos:
                for idx, c2 in get_A_entry_linear_comb(L_expr, r, c):
                    u_dict[idx] = u_dict.get(idx, 0) + coeff * c2
            
            # Build V vector
            v_dict = {}
            for r, c, coeff in R_pos:
                for idx, c2 in get_B_entry_linear_comb(R_expr, r, c):
                    v_dict[idx] = v_dict.get(idx, 0) + coeff * c2
            
            # Build W vector
            w_dict = {}
            for (mr, mc), alpha in result_coeffs[s].items():
                for (br, bc), beta in block_result_coeffs[j].items():
                    # mr, mc is position within block Mj
                    # br, bc is block index in C
                    # C_row = 2*br + mr
                    # C_col = 2*bc + mc
                    C_row = 2 * br + mr
                    C_col = 2 * bc + mc
                    W_idx = C_col * 4 + C_row
                    total_coeff = alpha * beta
                    w_dict[W_idx] = w_dict.get(W_idx, 0) + total_coeff
            
            # Fill U, V, W
            for idx in range(16):
                U[idx][col] = u_dict.get(idx, 0)
                V[idx][col] = v_dict.get(idx, 0)
                W[idx][col] = w_dict.get(idx, 0)
            
            col += 1
            
    return U, V, W


def construct():
    """
    Return a rank-minimal exact tensor decomposition
    for 4x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Use the rank-49 Strassen-based decomposition
    try:
        U, V, W = _strassen_4x4()
        return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook rank 64
    rank = 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
