import time

def construct():
    """
    Construct a rank-49 tensor decomposition for 4x4 matrix multiplication.
    Uses 2-level Strassen algorithm.
    Indices follow the trace convention:
    U: (i, j) -> i*4 + j
    V: (j, k) -> j*4 + k
    W: (k, i) -> k*4 + i
    """
    
    # Define Strassen scalar terms for 2x2 matrix multiplication
    # Each term is (x_indices, y_indices, z_indices)
    # x_indices: list of (r, c, coeff) for input X
    # y_indices: list of (r, c, coeff) for input Y
    # z_indices: list of (r, c, coeff) for output Z
    # X, Y, Z are 2x2 matrices.
    
    # Standard Strassen for 2x2:
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    #
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # We represent each scalar multiplication as a tuple:
    # (x_form, y_form, z_contrib)
    # x_form: dict {(r,c): coeff}
    # y_form: dict {(r,c): coeff}
    # z_contrib: dict {(r,c): coeff} indicating which Z entries get this scalar
    
    strassen_terms = [
        {
            'x': {(0,0): 1, (1,1): 1},
            'y': {(0,0): 1, (1,1): 1},
            'z': {(0,0): 1, (1,1): 1}
        },
        {
            'x': {(1,0): 1, (1,1): 1},
            'y': {(0,0): 1},
            'z': {(1,0): 1, (1,1): -1}
        },
        {
            'x': {(0,0): 1},
            'y': {(0,1): 1, (1,1): -1},
            'z': {(0,1): 1, (1,1): 1}
        },
        {
            'x': {(1,1): 1},
            'y': {(1,0): 1, (0,0): -1},
            'z': {(0,0): 1, (1,0): 1}
        },
        {
            'x': {(0,0): 1, (0,1): 1},
            'y': {(1,1): 1},
            'z': {(0,0): -1, (0,1): 1}
        },
        {
            'x': {(1,0): 1, (0,0): -1},
            'y': {(0,0): 1, (0,1): 1},
            'z': {(1,1): 1}
        },
        {
            'x': {(0,1): 1, (1,1): -1},
            'y': {(1,0): 1, (1,1): 1},
            'z': {(0,0): 1}
        }
    ]
    
    # Define the 7 block products for 4x4 using 2x2 blocks
    # Each block product k is defined by:
    # x_blocks: list of (block_r, block_c, coeff) for A
    # y_blocks: list of (block_r, block_c, coeff) for B
    # z_blocks: list of (block_r, block_c, coeff) for C
    # Note: The 'z' here means which C blocks receive the result of this block product.
    
    block_products = [
        {
            'x': [((0,0), 1), ((1,1), 1)],
            'y': [((0,0), 1), ((1,1), 1)],
            'z': [((0,0), 1), ((1,1), 1)]
        },
        {
            'x': [((1,0), 1), ((1,1), 1)],
            'y': [((0,0), 1)],
            'z': [((1,0), 1), ((1,1), -1)]
        },
        {
            'x': [((0,0), 1)],
            'y': [((0,1), 1), ((1,1), -1)],
            'z': [((0,1), 1), ((1,1), 1)]
        },
        {
            'x': [((1,1), 1)],
            'y': [((1,0), 1), ((0,0), -1)],
            'z': [((0,0), 1), ((1,0), 1)]
        },
        {
            'x': [((0,0), 1), ((0,1), 1)],
            'y': [((1,1), 1)],
            'z': [((0,0), -1), ((0,1), 1)]
        },
        {
            'x': [((1,0), 1), ((0,0), -1)],
            'y': [((0,0), 1), ((0,1), 1)],
            'z': [((1,1), 1)]
        },
        {
            'x': [((0,1), 1), ((1,1), -1)],
            'y': [((1,0), 1), ((1,1), 1)],
            'z': [((0,0), 1)]
        }
    ]
    
    rank = 49
    N = 16
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(N)]
    W = [[0] * rank for _ in range(N)]
    
    def block_to_flat(br, bc, r, c):
        """
        Convert block index (br, bc) and entry (r, c) to flat index in 4x4 matrix.
        br, bc in {0, 1}. r, c in {0, 1}.
        Row = 2*br + r
        Col = 2*bc + c
        Flat index (row-major) = Row * 4 + Col
        """
        row = 2 * br + r
        col = 2 * bc + c
        return row * 4 + col
    
    r_idx = 0
    for bp_idx, bp in enumerate(block_products):
        for st_idx, st in enumerate(strassen_terms):
            # 1. Build U (linear form on A)
            # st['x'] gives the 2x2 form for the X matrix of the block product.
            # bp['x'] gives the block composition of X.
            # U_flat[idx] += sum over x_entries and block_refs
            
            u_form = {}
            for (xr, xc, xcoeff) in [(k[0], k[1], v) for k, v in st['x'].items()]:
                # For each entry in the 2x2 X matrix
                for (br, bc, bcoeff) in bp['x']:
                    # For each block contributing to X
                    # The entry (xr, xc) of X is formed by summing (xr, xc) of the blocks
                    # Wait, the block product X is a 2x2 matrix.
                    # X[xr, xc] = sum_{(br,bc)} bcoeff * A_block[br, bc][xr, xc]
                    # So for a specific scalar term in Strassen, we pick X[xr, xc].
                    # The coefficient is xcoeff.
                    # So we add xcoeff * bcoeff to A_block[br, bc][xr, xc].
                    flat_idx = block_to_flat(br, bc, xr, xc)
                    u_form[flat_idx] = u_form.get(flat_idx, 0) + xcoeff * bcoeff
            
            # 2. Build V (linear form on B)
            v_form = {}
            for (yr, yc, ycoeff) in [(k[0], k[1], v) for k, v in st['y'].items()]:
                for (br, bc, bcoeff) in bp['y']:
                    flat_idx = block_to_flat(br, bc, yr, yc)
                    v_form[flat_idx] = v_form.get(flat_idx, 0) + ycoeff * bcoeff
            
            # 3. Build W (linear form on C)
            # The scalar product contributes to Z[yr, yc] of the block product M_k.
            # st['z'] tells us which entries of the 2x2 result Z get this scalar.
            # bp['z'] tells us which C blocks get the result Z.
            # Specifically, Z[yr, yc] contributes to C_block[cb_r, cb_c] with coeff zcoeff_bp.
            # And the entry Z[yr, yc] is at position (yr, yc) within the block.
            # So it contributes to C[2*cb_r + yr, 2*cb_c + yc].
            
            w_form = {}
            for (zr, zc, zcoeff_st) in [(k[0], k[1], v) for k, v in st['z'].items()]:
                # For each entry in the 2x2 result Z
                for (cb_r, cb_c, zcoeff_bp) in bp['z']:
                    # For each C block receiving the result
                    # The entry Z[zr, zc] maps to C[2*cb_r + zr, 2*cb_c + zc]
                    row = 2 * cb_r + zr
                    col = 2 * cb_c + zc
                    # Standard C index is (row, col).
                    # Trace convention for W is (k, i) where C_{ki} is the entry.
                    # In our derivation: C_{ba} corresponds to W index (b, a).
                    # Here C entry is at (row, col).
                    # So b = col, a = row.
                    # W index is (b, a) = (col, row).
                    # Flat index for W: k*4 + i = col*4 + row.
                    w_flat_idx = col * 4 + row
                    w_form[w_flat_idx] = w_form.get(w_flat_idx, 0) + zcoeff_st * zcoeff_bp
            
            # Fill U, V, W
            for idx, val in u_form.items():
                U[idx][r_idx] = val
            for idx, val in v_form.items():
                V[idx][r_idx] = val
            for idx, val in w_form.items():
                W[idx][r_idx] = val
            
            r_idx += 1
            
    return {"U": U, "V": V, "W": W}
