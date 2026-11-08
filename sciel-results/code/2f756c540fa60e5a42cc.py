import time
import sys

def construct():
    """
    Construct a rank-49 tensor decomposition for 4x4 matrix multiplication
    using 2-level Strassen (block Strassen + scalar Strassen).
    
    The decomposition uses:
    - 7 Strassen products at the block level (2x2 blocks of 2x2 matrices)
    - 7 Strassen products at the scalar level for each block product
    - Total rank = 7 * 7 = 49
    
    Index convention:
    - A, B are 4x4 matrices with flat index i*4+j for row i, col j
    - C = A*B with W[k*4+i, r] where k=col, i=row (trace convention)
    """
    rank = 49
    N = 16  # 4*4
    
    # Initialize U, V, W with zeros
    # U shape: (16, 49)
    # V shape: (16, 49)
    # W shape: (16, 49)
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(N)]
    W = [[0] * rank for _ in range(N)]
    
    # Block index helper: A[br*2+r, bc*2+s] flat index
    def block_idx(br, bc, r, s):
        return (br * 2 + r) * 4 + (bc * 2 + s)
    
    # 2x2 Strassen scalar terms
    # Each term: (x_entries, y_entries, z_contrib)
    # x_entries: list of (row, col, coeff)
    # y_entries: list of (row, col, coeff)
    # z_contrib: dict {(zr, zs): coeff}
    
    strassen_2x2 = [
        # M1 = (X00+X11)(Y00+Y11) -> Z00+=1, Z11+=1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], {(0,0):1, (1,1):1}),
        # M2 = (X10+X11)Y00 -> Z10+=1, Z11-=1
        ([(1,0,1),(1,1,1)], [(0,0,1)], {(1,0):1, (1,1):-1}),
        # M3 = X00(Y01-Y11) -> Z01+=1, Z11+=1
        ([(0,0,1)], [(0,1,1),(1,1,-1)], {(0,1):1, (1,1):1}),
        # M4 = X11(Y10-Y00) -> Z00+=1, Z10+=1
        ([(1,1,1)], [(1,0,1),(0,0,-1)], {(0,0):1, (1,0):1}),
        # M5 = (X00+X01)Y11 -> Z00-=1, Z01+=1
        ([(0,0,1),(0,1,1)], [(1,1,1)], {(0,0):-1, (0,1):1}),
        # M6 = (X10-X00)(Y00+Y01) -> Z11+=1
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], {(1,1):1}),
        # M7 = (X01-X11)(Y10+Y11) -> Z00+=1
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], {(0,0):1}),
    ]
    
    # Block-level Strassen: 7 products of 2x2 blocks
    # Each product: (x_blocks, y_blocks, c_map)
    # x_blocks: list of (br, bc, coeff)
    # y_blocks: list of (br, bc, coeff)
    # c_map: dict {(cb_r, cb_c): coeff}
    
    block_strassen = [
        # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], {(0,0):1, (1,1):1}),
        # M2 = (A10+A11)B00 -> C10+=1, C11-=1
        ([(1,0,1),(1,1,1)], [(0,0,1)], {(1,0):1, (1,1):-1}),
        # M3 = A00(B01-B11) -> C01+=1, C11+=1
        ([(0,0,1)], [(0,1,1),(1,1,-1)], {(0,1):1, (1,1):1}),
        # M4 = A11(B10-B00) -> C00+=1, C10+=1
        ([(1,1,1)], [(1,0,1),(0,0,-1)], {(0,0):1, (1,0):1}),
        # M5 = (A00+A01)B11 -> C00-=1, C01+=1
        ([(0,0,1),(0,1,1)], [(1,1,1)], {(0,0):-1, (0,1):1}),
        # M6 = (A10-A00)(B00+B01) -> C11+=1
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], {(1,1):1}),
        # M7 = (A01-A11)(B10+B11) -> C00+=1
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], {(0,0):1}),
    ]
    
    r_idx = 0
    for bprod_idx, (x_blocks, y_blocks, c_map) in enumerate(block_strassen):
        for sprod_idx, (x_entries, y_entries, z_contrib) in enumerate(strassen_2x2):
            # Build U column
            for flat_i in range(N):
                u_val = 0
                for (er, es, ecoeff) in x_entries:
                    for (br, bc, bcoeff) in x_blocks:
                        a_idx = block_idx(br, bc, er, es)
                        if a_idx == flat_i:
                            u_val += ecoeff * bcoeff
                U[flat_i][r_idx] = u_val
            
            # Build V column
            for flat_j in range(N):
                v_val = 0
                for (er, es, ecoeff) in y_entries:
                    for (br, bc, bcoeff) in y_blocks:
                        b_idx = block_idx(br, bc, er, es)
                        if b_idx == flat_j:
                            v_val += ecoeff * bcoeff
                V[flat_j][r_idx] = v_val
            
            # Build W column
            for w_idx in range(N):
                # w_idx = col*4 + row (trace convention)
                col = w_idx // 4
                row = w_idx % 4
                w_val = 0
                for (zr, zs), zcoeff in z_contrib.items():
                    for (cb_r, cb_c), ccoeff in c_map.items():
                        # Output C[cb_r*2+zr, cb_c*2+zs]
                        out_row = cb_r * 2 + zr
                        out_col = cb_c * 2 + zs
                        out_w_idx = out_col * 4 + out_row
                        if out_w_idx == w_idx:
                            w_val += zcoeff * ccoeff
                W[w_idx][r_idx] = w_val
            
            r_idx += 1
    
    return {"U": U, "V": V, "W": W}
