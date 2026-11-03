import time

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    block_row = {0: 0, 1: 0, 2: 1, 3: 1}
    block_col = {0: 0, 1: 1, 2: 0, 3: 1}
    
    block_defs = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
    strassen_defs = [
        ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)]),
        ([((1,0),1), ((1,1),1)], [((0,0),1)]),
        ([((0,0),1)], [((0,1),1), ((1,1),-1)]),
        ([((1,1),1)], [((1,0),1), ((0,0),-1)]),
        ([((0,0),1), ((0,1),1)], [((1,1),1)]),
        ([((1,0),1), ((0,0),-1)], [((0,0),1), ((0,1),1)]),
        ([((0,1),1), ((1,1),-1)], [((1,0),1), ((1,1),1)]),
    ]
    
    c_block_coeffs = {
        0: {0: 1, 1: 0, 2: 0, 3: 1},
        1: {0: 0, 1: 0, 2: 1, 3: -1},
        2: {0: 0, 1: 1, 2: 0, 3: 1},
        3: {0: 1, 1: 0, 2: 1, 3: 0},
        4: {0: -1, 1: 1, 2: 0, 3: 1},
        5: {0: 0, 1: 0, 2: 0, 3: 1},
        6: {0: 1, 1: 0, 2: 0, 3: 0},
    }
    
    z_entry_coeffs = {
        0: {0: 1, 1: 0, 2: 0, 3: 1},
        1: {0: 0, 1: 0, 2: 1, 3: -1},
        2: {0: 0, 1: 1, 2: 0, 3: 1},
        3: {0: 1, 1: 0, 2: 1, 3: 0},
        4: {0: -1, 1: 1, 2: 0, 3: 1},
        5: {0: 0, 1: 0, 2: 0, 3: 1},
        6: {0: 1, 1: 0, 2: 0, 3: 0},
    }
    
    col = 0
    for bi in range(7):
        if time.time() > deadline:
            break
        X_blocks, Y_blocks = block_defs[bi]
        for si in range(7):
            X_entries, Y_entries = strassen_defs[si]
            
            for (r, c), coeff_x in X_entries:
                for block_idx, coeff_b in X_blocks.items():
                    gr = 2 * block_row[block_idx] + r
                    gc = 2 * block_col[block_idx] + c
                    U[gr * m + gc][col] += coeff_x * coeff_b
            
            for (r, c), coeff_y in Y_entries:
                for block_idx, coeff_b in Y_blocks.items():
                    gr = 2 * block_row[block_idx] + r
                    gc = 2 * block_col[block_idx] + c
                    V[gr * p + gc][col] += coeff_y * coeff_b
            
            for cblock_idx, c_coeff in c_block_coeffs[bi].items():
                if c_coeff == 0:
                    continue
                cbi = cblock_idx // 2
                cbj = cblock_idx % 2
                for zentry_idx, z_coeff in z_entry_coeffs[si].items():
                    if z_coeff == 0:
                        continue
                    zr = zentry_idx // 2
                    zc = zentry_idx % 2
                    gr = 2 * cbi + zr
                    gc = 2 * cbj + zc
                    W[gc * n + gr][col] += c_coeff * z_coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
