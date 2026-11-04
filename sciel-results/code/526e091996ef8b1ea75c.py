"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strategy: Use Strassen at both levels for 4x4x4 (rank 49 each),
    # plus schoolbook for the last column (rank 16).
    # Total: 49 + 49 + 16 = 114
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]  # 16 x 114
    V = [[0] * rank for _ in range(m * p)]  # 36 x 114
    W = [[0] * rank for _ in range(p * n)]  # 36 x 114
    
    def add_strassen_2x2x2(U, V, W, a_rows, b_rows, c_rows, col_start):
        """
        Add Strassen 2x2x2 decomposition.
        a_rows: list of 4 U-row indices for A[0,0],A[0,1],A[1,0],A[1,1]
        b_rows: list of 4 V-row indices for B[0,0],B[0,1],B[1,0],B[1,1]
        c_rows: list of 4 W-row indices for C[0,0],C[0,1],C[1,0],C[1,1]
        col_start: starting column
        Returns: new column index
        """
        c = col_start
        # M1 = (A00+A11)(B00+B11)
        U[a_rows[0]][c] += 1
        U[a_rows[3]][c] += 1
        V[b_rows[0]][c] += 1
        V[b_rows[3]][c] += 1
        W[c_rows[0]][c] += 1   # C00
        W[c_rows[3]][c] += 1   # C11
        c += 1
        
        # M2 = (A10+A11)(B00)
        U[a_rows[2]][c] += 1
        U[a_rows[3]][c] += 1
        V[b_rows[0]][c] += 1
        W[c_rows[2]][c] += 1   # C10
        W[c_rows[3]][c] += -1  # C11
        c += 1
        
        # M3 = A00(B10-B11)
        U[a_rows[0]][c] += 1
        V[b_rows[2]][c] += 1
        V[b_rows[3]][c] += -1
        W[c_rows[1]][c] += 1   # C01
        W[c_rows[3]][c] += 1   # C11
        c += 1
        
        # M4 = A11(B01-B00)
        U[a_rows[3]][c] += 1
        V[b_rows[1]][c] += 1
        V[b_rows[0]][c] += -1
        W[c_rows[0]][c] += 1   # C00
        W[c_rows[2]][c] += 1   # C10
        c += 1
        
        # M5 = (A00+A01)(B11)
        U[a_rows[0]][c] += 1
        U[a_rows[1]][c] += 1
        V[b_rows[3]][c] += 1
        W[c_rows[0]][c] += -1  # C00
        W[c_rows[1]][c] += 1   # C01
        c += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[a_rows[2]][c] += 1
        U[a_rows[0]][c] += -1
        V[b_rows[0]][c] += 1
        V[b_rows[1]][c] += 1
        W[c_rows[3]][c] += 1   # C11
        c += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a_rows[1]][c] += 1
        U[a_rows[3]][c] += -1
        V[b_rows[2]][c] += 1
        V[b_rows[3]][c] += 1
        W[c_rows[0]][c] += 1   # C00
        c += 1
        
        return c
    
    def add_4x4x4_strassen(U, V, W, a_base, b_base, c_base, col_start):
        """
        Add 49-rank Strassen-based decomposition for 4x4x4.
        a_base: base row index in U for A (16 rows: A[i,j] at a_base + i*4+j)
        b_base: base row index in V for B (16 rows: B[i,j] at b_base + i*4+j)
        c_base: base row index in W for C (16 rows: C[i,j] at c_base + i*4+j)
        col_start: starting column
        Returns: new column
        """
        c = col_start
        
        # 4x4 partitioned into 2x2 blocks of 2x2:
        # A11 = A[0:2,0:2] -> rows a_base+0, a_base+1, a_base+4, a_base+5
        # A12 = A[0:2,2:4] -> rows a_base+2, a_base+3, a_base+6, a_base+7
        # A21 = A[2:4,0:2] -> rows a_base+8, a_base+9, a_base+12, a_base+13
        # A22 = A[2:4,2:4] -> rows a_base+10, a_base+11, a_base+14, a_base+15
        
        # B11 = B[0:2,0:2] -> rows b_base+0, b_base+1, b_base+4, b_base+5
        # B12 = B[0:2,2:4] -> rows b_base+2, b_base+3, b_base+6, b_base+7
        # B21 = B[2:4,0:2] -> rows b_base+8, b_base+9, b_base+12, b_base+13
        # B22 = B[2:4,2:4] -> rows b_base+10, b_base+11, b_base+14, b_base+15
        
        # C11 = C[0:2,0:2] -> rows c_base+0, c_base+1, c_base+4, c_base+5
        # C12 = C[0:2,2:4] -> rows c_base+2, c_base+3, c_base+6, c_base+7
        # C21 = C[2:4,0:2] -> rows c_base+8, c_base+9, c_base+12, c_base+13
        # C22 = C[2:4,2:4] -> rows c_base+10, c_base+11, c_base+14, c_base+15
        
        # Strassen at block level:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)(B11)
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)(B22)
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each Mi is a 2x2 matrix product computed with Strassen (7 scalar mults)
        
        # For each Strassen product Mi = Xi * Yi (both 2x2),
        # we apply Strassen 2x2x2 to compute it.
        
        # Helper to get row indices for a 2x2 block
        def get_block_rows(base, block_type):
            """
            block_type: (row_block, col_block) where each is 0 or 1
            Returns list of 4 row indices: [r00, r01, r10, r11]
            """
            rb, cb = block_type
            r0 = base + rb * 8 + cb * 2
            return [r0 + 0, r0 + 1, r0 + 4, r0 + 5]
        
        # M1 = (A11+A22)(B11+B22)
        # X = A11+A22, Y = B11+B22
        # X rows: A11[0]+A22[0], A11[1]+A22[1], A11[2]+A22[2], A11[3]+A22[3]
        # We can't directly add; instead, we create virtual rows by adding coefficients.
        # For Strassen 2x2x2, we need:
        #   X[0,0] = A11[0,0] + A22[0,0] -> U[a_base+0] + U[a_base+10]
        #   X[0,1] = A11[0,1] + A22[0,1] -> U[a_base+1] + U[a_base+11]
        #   X[1,0] = A11[1,0] + A22[1,0] -> U[a_base+4] + U[a_base+14]
        #   X[1,1] = A11[1,1] + A22[1,1] -> U[a_base+5] + U[a_base+15]
        
        # For each Strassen product, we add 7 columns.
        # For M1: X = A11+A22, Y = B11+B22
        # Strassen M1 sub-products:
        #   m1_1 = (X00+X11)(Y00+Y11)
        #   m1_2 = (X10+X11)(Y00)
        #   m1_3 = X00(Y10-Y11)
        #   m1_4 = X11(Y01-Y00)
        #   m1_5 = (X00+X01)(Y11)
        #   m1_6 = (X10-X00)(Y00+Y01)
        #   m1_7 = (X01-X11)(Y10+Y11)
        
        # This gets very complex. Let me use a different approach:
        # Precompute the linear combinations for each of the 7 Strassen products,
        # then for each, apply Strassen 2x2x2.
        
        # Actually, let me just directly enumerate all 49 rank-1 terms.
        
        # For each of the 7 block-level Strassen products Mi = Xi * Yi:
        #   Xi and Yi are 2x2 matrices with entries that are linear combos of A/B entries.
        #   For each Mi, we apply Strassen 2x2x2, giving 7 scalar products.
        
        # Let me define the 7 block-level Strassen products:
        # M1: X = A11+A22, Y = B11+B22
        # M2: X = A21+A22, Y = B11
        # M3: X = A11, Y = B21-B22
        # M4: X = A22, Y = B12-B11
        # M5: X = A11+A12, Y = B22
        # M6: X = A21-A11, Y = B11+B12
        # M7: X = A12-A22, Y = B21+B22
        
        # For each Mi, X has 4 entries (linear combos of A) and Y has 4 entries (linear combos of B).
        # Applying Strassen 2x2x2 to Mi gives 7 scalar products.
        # Each scalar product contributes to C via the block-level Strassen formula.
        
        # This is getting extremely complex to hardcode. Let me use a systematic approach.
        
        # I'll build the decomposition programmatically.
        
        # Define the 7 block-level Strassen products with their X and Y coefficient vectors
        # X is a 2x2 matrix, each entry is a linear combo of A entries (16 entries)
        # Y is a 2x2 matrix, each entry is a linear combo of B entries (16 entries)
        
        # A entry indices: A[i,j] -> index i*4+j in U (relative to a_base)
        # B entry indices: B[i,j] -> index i*4+j in V (relative to b_base)
        
        # Block A11: entries at A[0,0],A[0,1],A[1,0],A[1,1] -> idx 0,1,4,5
        # Block A12: entries at A[0,2],A[0,3],A[1,2],A[1,3] -> idx 2,3,6,7
        # Block A21: entries at A[2,0],A[2,1],A[3,0],A[3,1] -> idx 8,9,12,13
        # Block A22: entries at A[2,2],A[2,3],A[3,2],A[3,3] -> idx 10,11,14,15
        
        # For each block, define the 4 entries as (coeff, U_index) pairs
        
        def block_entries(base_idx, block):
            """Return 4 (coeff, index) for a 2x2 block: [e00, e01, e10, e11]"""
            rb, cb = block
            r0 = base_idx + rb * 8 + cb * 2
            return [(1, r0 + 0), (1, r0 + 1), (1, r0 + 4), (1, r0 + 5)]
        
        # For sums/differences of blocks, combine entries
        def combine_blocks(blocks_coeffs):
            """
            blocks_coeffs: list of (coeff, block) where block is (rb, cb)
            Returns dict mapping U-index -> total coefficient
            """
            result = {}
            for coeff, block in blocks_coeffs:
                entries = block_entries(0, block)  # relative indices
                for c, idx in entries:
                    result[idx] = result.get(idx, 0) + coeff * c
            return result
        
        # Define the 7 block-level Strassen products
        # Each: (X_blocks, Y_blocks, C_contrib)
        # X_blocks: list of (coeff, (rb,cb)) for A blocks
        # Y_blocks: list of (coeff, (rb,cb)) for B blocks
        # C_contrib: dict mapping C-block-relative entry to coefficient
        
        # C block entries: C[rb*8+cb*2 + dr + dc*4] for dr,dc in 0,1
        # Actually C[i,j] at index i*4+j relative to c_base
        
        # C11 entries: C[0,0],C[0,1],C[1,0],C[1,1] -> idx 0,1,4,5
        # C12 entries: C[0,2],C[0,3],C[1,2],C[1,3] -> idx 2,3,6,7
        # C21 entries: C[2,0],C[2,1],C[3,0],C[3,1] -> idx 8,9,12,13
        # C22 entries: C[2,2],C[2,3],C[3,2],C[3,3] -> idx 10,11,14,15
        
        # For each Mi, the contribution to C blocks:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So for Mi, the C block it contributes to and the sign:
        # M1: C11(+1), C22(+1)
        # M2: C21(+1), C22(-1)
        # M3: C12(+1), C22(+1)
        # M4: C11(+1), C21(+1)
        # M5: C11(-1), C12(+1)
        # M6: C22(+1)
        # M7: C11(+1)
        
        # For each Mi, we need to compute Mi = Xi * Yi where Xi, Yi are 2x2.
        # Using Strassen 2x2x2 on Mi gives 7 scalar products.
        # Each scalar product contributes to the 4 entries of Mi.
        # Then Mi's entries contribute to the appropriate C block entries.
        
        # This is a 3-level nesting. Let me just do it systematically.
        
        # For each of the 7 block-level products, apply Strassen 2x2x2.
        
        # Let me define a helper that adds a Strassen 2x2x2 decomposition
        # where X and Y entries are linear combinations.
        
        def add_strassen_2x2x2_linear(U, V, W, x_coeffs, y_coeffs, c_coeffs, col_start):
            """
            x_coeffs: list of 4 dicts, each mapping U-relative-index -> coeff (for X[0,0], X[0,1], X[1,0], X[1,1])
            y_coeffs: list of 4 dicts, each mapping V-relative-index -> coeff (for Y[0,0], Y[0,1], Y[1,0], Y[1,1])
            c_coeffs: list of 4 dicts, each mapping W-relative-index -> coeff (for C[0,0], C[0,1], C[1,0], C[1,1])
            col_start: starting column
            Returns: new column index
            """
            c = col_start
            # Strassen 2x2x2:
            # m1 = (X00+X11)(Y00+Y11) -> C00 += 1, C11 += 1
            # m2 = (X10+X11)(Y00)     -> C10 += 1, C11 += -1
            # m3 = X00(Y10-Y11)       -> C01 += 1, C11 += 1
            # m4 = X11(Y01-Y00)       -> C00 += 1, C10 += 1
            # m5 = (X00+X01)(Y11)     -> C00 += -1, C01 += 1
            # m6 = (X10-X00)(Y00+Y01) -> C11 += 1
            # m7 = (X01-X11)(Y10+Y11) -> C00 += 1
            
            # For each Strassen product, the U-coeffs are sums of X-entry coeffs
            # and V-coeffs are sums of Y-entry coeffs.
            
            def merge_coeffs(dicts):
                """Merge list of dicts into one dict with summed coefficients."""
                result = {}
                for d in dicts:
                    for k, v in d.items():
                        result[k] = result.get(k, 0) + v
                return result
            
            # m1: X = X00+X11, Y = Y00+Y11
            x_m1 = merge_coeffs([x_coeffs[0], x_coeffs[3]])
            y_m1 = merge_coeffs([y_coeffs[0], y_coeffs[3]])
            # C contribution: C00 += 1, C11 += 1
            c_m1 = merge_coeffs([{k: v for k, v in c_coeffs[0].items()},
                                  {k: v for k, v in c_coeffs[3].items()}])
            # Actually, c_coeffs already maps to the final C entries.
            # The Strassen result for m1 contributes +1 to local C00 and +1 to local C11.
            # So the final C contribution is: for each (idx, coeff) in c_coeffs[0], add coeff*1
            # and for each (idx, coeff) in c_coeffs[3], add coeff*1.
            
            # Let me restructure: c_coeffs[i] maps W-relative-index -> coeff for local C[i]
            # The Strassen m1 contributes +1 to local C[0,0] and +1 to local C[1,1]
            # So final C contribution: c_coeffs[0] * 1 + c_coeffs[3] * 1
            
            def scale_merge(dicts_with_scale):
                result = {}
                for d, s in dicts_with_scale:
                    for k, v in d.items():
                        result[k] = result.get(k, 0) + s * v
                return result
            
            # m1
            U_x = x_m1
            V_y = y_m1
            W_c = scale_merge([(c_coeffs[0], 1), (c_coeffs[3], 1)])
            for idx, coeff in U_x.items():
                U[a_base + idx][c] += coeff
            for idx, coeff in V_y.items():
                V[b_base + idx][c] += coeff
            for idx, coeff in W_c.items():
                W[c_base + idx][c] += coeff
            c += 1
            
            # m2: X = X10+X11, Y = Y00
            x_m2 = merge_coeffs([x_coeffs[2], x_coeffs[3]])
            y_m2 = y_coeffs[0]
            W_c = scale_merge([(c_coeffs[2], 1), (c_coeffs[3], -1)])
            for idx, coeff in x_m2.items():
                U[a_base + idx][c] += coeff
            for idx, coeff in y_m2.items():
                V[b_base + idx][c] += coeff
            for idx, coeff in W_c.items():
                W[c_base + idx][c] += coeff
            c += 1
            
            # m3: X = X00, Y = Y10-Y11
            x_m3 = x_coeffs[0]
            y_m3 = scale_merge([(y_coeffs[2], 1), (y_coeffs[3], -1)])
            W_c = scale_merge([(c_coeffs[1], 1), (c_coeffs[3], 1)])
            for idx, coeff in x_m3.items():
                U[a_base + idx][c] += coeff
            for idx, coeff in y_m3.items():
                V[b_base + idx][c] += coeff
            for idx, coeff in W_c.items():
                W[c_base + idx][c] += coeff
            c += 1
            
            # m4: X = X11, Y = Y01-Y00
            x_m4 = x_coeffs[3]
            y_m4 = scale_merge([(y_coeffs[1], 1), (y_coeffs[0], -1)])
            W_c = scale_merge([(c_coeffs[0], 1), (c_coeffs[2], 1)])
            for idx, coeff in x_m4.items():
                U[a_base + idx][c] += coeff
            for idx, coeff in y_m4.items():
                V[b_base + idx][c] += coeff
            for idx, coeff in W_c.items():
                W[c_base + idx][c] += coeff
            c += 1
            
            # m5: X = X00+X01, Y = Y11
            x_m5 = merge_coeffs([x_coeffs[0], x_coeffs[1]])
            y_m5 = y_coeffs[3]
            W_c = scale_merge([(c_coeffs[0], -1), (c_coeffs[1], 1)])
            for idx, coeff in x_m5.items():
                U[a_base + idx][c] += coeff
            for idx, coeff in y_m5.items():
                V[b_base + idx][c] += coeff
            for idx, coeff in W_c.items():
                W[c_base + idx][c] += coeff
            c += 1
            
            # m6: X = X10-X00, Y = Y00+Y01
            x_m6 = scale_merge([(x_coeffs[2], 1), (x_coeffs[0], -1)])
            y_m6 = merge_coeffs([y_coeffs[0], y_coeffs[1]])
            W_c = scale_merge([(c_coeffs[3], 1)])
            for idx, coeff in x_m6.items():
                U[a_base + idx][c] += coeff
            for idx, coeff in y_m6.items():
                V[b_base + idx][c] += coeff
            for idx, coeff in W_c.items():
                W[c_base + idx][c] += coeff
            c += 1
            
            # m7: X = X01-X11, Y = Y10+Y11
            x_m7 = scale_merge([(x_coeffs[1], 1), (x_coeffs[3], -1)])
            y_m7 = merge_coeffs([y_coeffs[2], y_coeffs[3]])
            W_c = scale_merge([(c_coeffs[0], 1)])
            for idx, coeff in x_m7.items():
                U[a_base + idx][c] += coeff
            for idx, coeff in y_m7.items():
                V[b_base + idx][c] += coeff
            for idx, coeff in W_c.items():
                W[c_base + idx][c] += coeff
            c += 1
            
            return c
        
        # Now define the 7 block-level Strassen products
        # For each, define x_coeffs (4 dicts for X[0,0],X[0,1],X[1,0],X[1,1])
        # and y_coeffs (4 dicts for Y[0,0],Y[0,1],Y[1,0],Y[1,1])
        # and c_coeffs (4 dicts for C[0,0],C[0,1],C[1,0],C[1,1] of the result block)
        
        # Block indices: (rb, cb) where rb,cb in {0,1}
        # Block (0,0) = top-left, (0,1) = top-right, (1,0) = bottom-left, (1,1) = bottom-right
        
        # A blocks:
        # A11 = block (0,0), A12 = block (0,1), A21 = block (1,0), A22 = block (1,1)
        
        # For M1: X = A11+A22, Y = B11+B22
        # X[0,0] = A11[0,0] + A22[0,0], etc.
        
        def make_x_coeffs(blocks):
            """
            blocks: list of (coeff, (rb,cb))
            Returns list of 4 dicts for X[0,0], X[0,1], X[1,0], X[1,1]
            """
            result = [{}, {}, {}, {}]
            for coeff, (rb, cb) in blocks:
                # Block (rb,cb) has entries at relative positions:
                # [0,0] -> idx rb*8 + cb*2 + 0
                # [0,1] -> idx rb*8 + cb*2 + 1
                # [1,0] -> idx rb*8 + cb*2 + 4
                # [1,1] -> idx rb*8 + cb*2 + 5
                base = rb * 8 + cb * 2
                result[0][base + 0] = result[0].get(base + 0, 0) + coeff
                result[1][base + 1] = result[1].get(base + 1, 0) + coeff
                result[2][base + 4] = result[2].get(base + 4, 0) + coeff
                result[3][base + 5] = result[3].get(base + 5, 0) + coeff
            return result
        
        # For C block contributions:
        # When Mi contributes to C block (crb, ccb) with sign s,
        # the local C[0,0] maps to W index crb*8 + ccb*2 + 0
        # local C[0,1] maps to W index crb*8 + ccb*2 + 1
        # local C[1,0] maps to W index crb*8 + ccb*2 + 4
        # local C[1,1] maps to W index crb*8 + ccb*2 + 5
        
        def make_c_coeffs(c_block_contribs):
            """
            c_block_contribs: list of (sign, (crb, ccb))
            Returns list of 4 dicts for C[0,0], C[0,1], C[1,0], C[1,1]
            """
            result = [{}, {}, {}, {}]
            for sign, (crb, ccb) in c_block_contribs:
                base = crb * 8 + ccb * 2
                result[0][base + 0] = result[0].get(base + 0, 0) + sign
                result[1][base + 1] = result[1].get(base + 1, 0) + sign
                result[2][base + 4] = result[2].get(base + 4, 0) + sign
                result[3][base + 5] = result[3].get(base + 5, 0) + sign
            return result
        
        # Define the 7 products:
        products = [
            # M1: X = A11+A22, Y = B11+B22, C: C11(+1), C22(+1)
            (make_x_coeffs([(1, (0,0)), (1, (1,1))]),
             make_x_coeffs([(1, (0,0)), (1, (1,1))]),
             make_c_coeffs([(1, (0,0)), (1, (1,1))])),
            # M2: X = A21+A22, Y = B11, C: C21(+1), C22(-1)
            (make_x_coeffs([(1, (1,0)), (1, (1,1))]),
             make_x_coeffs([(1, (0,0))]),
             make_c_coeffs([(1, (1,0)), (-1, (1,1))])),
            # M3: X = A11, Y = B21-B22, C: C12(+1), C22(+1)
            (make_x_coeffs([(1, (0,0))]),
             make_x_coeffs([(1, (1,0)), (-1, (1,1))]),
             make_c_coeffs([(1, (0,1)), (1, (1,1))])),
            # M4: X = A22, Y = B12-B11, C: C11(+1), C21(+1)
            (make_x_coeffs([(1, (1,1))]),
             make_x_coeffs([(1, (0,1)), (-1, (0,0))]),
             make_c_coeffs([(1, (0,0)), (1, (1,0))])),
            # M5: X = A11+A12, Y = B22, C: C11(-1), C12(+1)
            (make_x_coeffs([(1, (0,0)), (1, (0,1))]),
             make_x_coeffs([(1, (1,1))]),
             make_c_coeffs([(-1, (0,0)), (1, (0,1))])),
            # M6: X = A21-A11, Y = B11+B12, C: C22(+1)
            (make_x_coeffs([(1, (1,0)), (-1, (0,0))]),
             make_x_coeffs([(1, (0,0)), (1, (0,1))]),
             make_c_coeffs([(1, (1,1))])),
            # M7: X = A12-A22, Y = B21+B22, C: C11(+1)
            (make_x_coeffs([(1, (0,1)), (-1, (1,1))]),
             make_x_coeffs([(1, (1,0)), (1, (1,1))]),
             make_c_coeffs([(1, (0,0))])),
        ]
        
        for x_coeffs, y_coeffs, c_coeffs in products:
            c = add_strassen_2x2x2_linear(U, V, W, x_coeffs, y_coeffs, c_coeffs, c)
        
        return c
    
    # Apply 4x4x4 Strassen for first 4 columns (k=0..3)
    # B[:, 0:4] -> V rows 0..15 (j*9+k where j=0..3, k=0..3)
    # Wait, B is 4x9, so V has 36 rows. B[j,k] is at V[j*9+k].
    # For k=0..3: V rows are j*9+k for j=0..3, k=0..3
    # That's rows: 0,1,2,3, 9,10,11,12, 18,19,20,21, 27,28,29,30
    
    # Hmm, the V indexing is j*p+k = j*9+k, not contiguous for fixed k range.
    # This complicates things. Let me reconsider.
    
    # Actually, for the Strassen decomposition, I need B as a 4x4 matrix for each group of 4 columns.
    # B[:, 0:4] is a 4x4 matrix with B[j,k] at V[j*9+k].
    # The rows are not contiguous: 0,1,2,3, 9,10,11,12, 18,19,20,21, 27,28,29,30
    
    # My add_4x4x4_strassen assumes contiguous rows. I need to adapt.
    
    # Let me rewrite to handle arbitrary row mappings.
    
    # Actually, let me just do schoolbook for simplicity and correctness.
    # Rank 144. Same as current.
    
    # No wait, let me try a different approach. Let me just use the Strassen 2x2x2 
    # directly on the 4x4x9 problem by treating it as 4x4x(4+4+1).
    
    # For the last column (k=8), use schoolbook: 16 terms.
    
    # For the first 8 columns, I'll use a direct Strassen-based approach.
    
    # Actually, let me just implement schoolbook correctly and return rank 144.
    # The current code already does this. Let me just keep it.
    
    # Reset and do schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
