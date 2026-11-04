"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 128 decomposition using block Strassen-like approach
    # 4 blocks of (2x2 x 2x9)
    # Each 2x2 x 2x9 is 4*(2x2x2) + 1*(2x2x1) = 4*7 + 4 = 32
    # Total 4 * 32 = 128
    
    rank = 128
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    def add_strassen_222(a_map, b_map, c_map):
        nonlocal current_rank
        # a_map: dict {logical_idx: global_U_idx}
        # b_map: dict {logical_idx: global_V_idx}
        # c_map: dict {logical_idx: global_W_idx}
        
        # Logical indices: 00, 01, 10, 11
        # 0: 00, 1: 01, 2: 10, 3: 11
        
        # M1 = (a00 + a11)(b00 + b11)
        u = {a_map[0]: 1, a_map[3]: 1}
        v = {b_map[0]: 1, b_map[3]: 1}
        w = {c_map[0]: 1, c_map[3]: -1} # C00 += 1, C11 += -1 ? No.
        # C00 = M1 - M2 + M3 - M5
        # C11 = M1 - M3 + M4 - M7
        # So M1 contributes +1 to C00 and +1 to C11.
        w = {c_map[0]: 1, c_map[3]: 1}
        for ui, val in u.items(): U[ui][current_rank] += val
        for vi, val in v.items(): V[vi][current_rank] += val
        for wi, val in w.items(): W[wi][current_rank] += val
        current_rank += 1
        
        # M2 = (a10 + a11)b00
        u = {a_map[2]: 1, a_map[3]: 1}
        v = {b_map[0]: 1}
        # C00 += -1, C10 += 1
        w = {c_map[0]: -1, c_map[2]: 1}
        for ui, val in u.items(): U[ui][current_rank] += val
        for vi, val in v.items(): V[vi][current_rank] += val
        for wi, val in w.items(): W[wi][current_rank] += val
        current_rank += 1
        
        # M3 = a00(b10 - b00)
        u = {a_map[0]: 1}
        v = {b_map[2]: 1, b_map[0]: -1}
        # C00 += 1, C11 += -1
        w = {c_map[0]: 1, c_map[3]: -1}
        for ui, val in u.items(): U[ui][current_rank] += val
        for vi, val in v.items(): V[vi][current_rank] += val
        for wi, val in w.items(): W[wi][current_rank] += val
        current_rank += 1
        
        # M4 = a11(b11 - b01)
        u = {a_map[3]: 1}
        v = {b_map[3]: 1, b_map[1]: -1}
        # C10 += 1, C11 += 1
        w = {c_map[2]: 1, c_map[3]: 1}
        for ui, val in u.items(): U[ui][current_rank] += val
        for vi, val in v.items(): V[vi][current_rank] += val
        for wi, val in w.items(): W[wi][current_rank] += val
        current_rank += 1
        
        # M5 = (a00 + a01)b11
        u = {a_map[0]: 1, a_map[1]: 1}
        v = {b_map[3]: 1}
        # C00 += -1, C01 += 1
        w = {c_map[0]: -1, c_map[1]: 1}
        for ui, val in u.items(): U[ui][current_rank] += val
        for vi, val in v.items(): V[vi][current_rank] += val
        for wi, val in w.items(): W[wi][current_rank] += val
        current_rank += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        u = {a_map[2]: 1, a_map[0]: -1}
        v = {b_map[0]: 1, b_map[1]: 1}
        # C01 += 1
        w = {c_map[1]: 1}
        for ui, val in u.items(): U[ui][current_rank] += val
        for vi, val in v.items(): V[vi][current_rank] += val
        for wi, val in w.items(): W[wi][current_rank] += val
        current_rank += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        u = {a_map[1]: 1, a_map[3]: -1}
        v = {b_map[2]: 1, b_map[3]: 1}
        # C11 += -1
        w = {c_map[3]: -1}
        for ui, val in u.items(): U[ui][current_rank] += val
        for vi, val in v.items(): V[vi][current_rank] += val
        for wi, val in w.items(): W[wi][current_rank] += val
        current_rank += 1

    def add_schoolbook_221(a_map, b_map, c_map):
        nonlocal current_rank
        # b_map: {0: b00, 1: b10}
        # c_map: {0: c00, 1: c10}
        
        # M1 = a00 b00
        u = {a_map[0]: 1}
        v = {b_map[0]: 1}
        w = {c_map[0]: 1}
        for ui, val in u.items(): U[ui][current_rank] += val
        for vi, val in v.items(): V[vi][current_rank] += val
        for wi, val in w.items(): W[wi][current_rank] += val
        current_rank += 1
        
        # M2 = a01 b10
        u = {a_map[1]: 1}
        v = {b_map[1]: 1}
        w = {c_map[0]: 1}
        for ui, val in u.items(): U[ui][current_rank] += val
        for vi, val in v.items(): V[vi][current_rank] += val
        for wi, val in w.items(): W[wi][current_rank] += val
        current_rank += 1
        
        # M3 = a10 b00
        u = {a_map[2]: 1}
        v = {b_map[0]: 1}
        w = {c_map[1]: 1}
        for ui, val in u.items(): U[ui][current_rank] += val
        for vi, val in v.items(): V[vi][current_rank] += val
        for wi, val in w.items(): W[wi][current_rank] += val
        current_rank += 1
        
        # M4 = a11 b10
        u = {a_map[3]: 1}
        v = {b_map[1]: 1}
        w = {c_map[1]: 1}
        for ui, val in u.items(): U[ui][current_rank] += val
        for vi, val in v.items(): V[vi][current_rank] += val
        for wi, val in w.items(): W[wi][current_rank] += val
        current_rank += 1

    # Define A block maps
    # A is 4x4, U index = i*4 + j
    # Block (bi, bj) covers rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    a_maps = {}
    for bi in range(2):
        for bj in range(2):
            r0 = 2 * bi
            c0 = 2 * bj
            a_maps[(bi, bj)] = {
                0: r0 * 4 + c0,       # 00
                1: r0 * 4 + c0 + 1,   # 01
                2: (r0 + 1) * 4 + c0, # 10
                3: (r0 + 1) * 4 + c0 + 1 # 11
            }

    # Define B block maps
    # B is 4x9, V index = i*9 + j
    # Block (bi) covers rows 2*bi..2*bi+1
    b_maps = {}
    for bi in range(2):
        r0 = 2 * bi
        b_maps[bi] = {
            # col c, row 0: r0*9 + c
            # col c, row 1: (r0+1)*9 + c
            # We need to pass specific columns in the call
            # So we will construct b_map on the fly for each column group
            'base_row0': r0 * 9,
            'base_row1': (r0 + 1) * 9
        }

    # Define C block maps
    # C is 4x9, W index = k*4 + i (k is col, i is row)
    # Block (ci) covers rows 2*ci..2*ci+1
    c_maps = {}
    for ci in range(2):
        r0 = 2 * ci
        c_maps[ci] = {
            # col c, row 0: c*4 + r0
            # col c, row 1: c*4 + r0 + 1
            'base_col_offset': 0 # handled in call
        }

    # Process each A block
    for bi in range(2):
        for bj in range(2):
            a_map = a_maps[(bi, bj)]
            b_block_idx = bj # B block index (0 for top, 1 for bottom)
            c_block_idx = bi # C block index (0 for top, 1 for bottom)
            
            b_base0 = b_maps[b_block_idx]['base_row0']
            b_base1 = b_maps[b_block_idx]['base_row1']
            
            # Process column groups
            col_groups = [(0, 1), (2, 3), (4, 5), (6, 7), (8,)]
            
            for group in col_groups:
                if len(group) == 2:
                    c1, c2 = group
                    # b_map for 2x2:
                    # 0: b00 (row 0, col c1) -> b_base0 + c1
                    # 1: b01 (row 0, col c2) -> b_base0 + c2
                    # 2: b10 (row 1, col c1) -> b_base1 + c1
                    # 3: b11 (row 1, col c2) -> b_base1 + c2
                    b_map = {
                        0: b_base0 + c1,
                        1: b_base0 + c2,
                        2: b_base1 + c1,
                        3: b_base1 + c2
                    }
                    # c_map for 2x2:
                    # 0: c00 (row 0, col c1) -> c1*4 + c_block_idx*2
                    # 1: c01 (row 0, col c2) -> c2*4 + c_block_idx*2
                    # 2: c10 (row 1, col c1) -> c1*4 + c_block_idx*2 + 1
                    # 3: c11 (row 1, col c2) -> c2*4 + c_block_idx*2 + 1
                    c_map = {
                        0: c1 * 4 + c_block_idx * 2,
                        1: c2 * 4 + c_block_idx * 2,
                        2: c1 * 4 + c_block_idx * 2 + 1,
                        3: c2 * 4 + c_block_idx * 2 + 1
                    }
                    add_strassen_222(a_map, b_map, c_map)
                else:
                    c1 = group[0]
                    # b_map for 2x1:
                    # 0: b00 (row 0, col c1) -> b_base0 + c1
                    # 1: b10 (row 1, col c1) -> b_base1 + c1
                    b_map = {
                        0: b_base0 + c1,
                        1: b_base1 + c1
                    }
                    # c_map for 2x1:
                    # 0: c00 (row 0, col c1) -> c1*4 + c_block_idx*2
                    # 1: c10 (row 1, col c1) -> c1*4 + c_block_idx*2 + 1
                    c_map = {
                        0: c1 * 4 + c_block_idx * 2,
                        1: c1 * 4 + c_block_idx * 2 + 1
                    }
                    add_schoolbook_221(a_map, b_map, c_map)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
