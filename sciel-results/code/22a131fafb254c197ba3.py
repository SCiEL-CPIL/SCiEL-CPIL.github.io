"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, use 115s for safety
    
    n, m, p = DIMENSIONS
    # Rank is 7 * 27 = 189
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block offsets for 3x3 blocks in 6x6 matrix
    # Block 11: rows 0-2, cols 0-2
    # Block 12: rows 0-2, cols 3-5
    # Block 21: rows 3-5, cols 0-2
    # Block 22: rows 3-5, cols 3-5
    offsets = {
        11: (0, 0),
        12: (0, 3),
        21: (3, 0),
        22: (3, 3)
    }
    
    # Strassen terms
    # l: (X_blocks, Y_blocks, targets)
    # X_blocks: dict {block_id: coeff}
    # Y_blocks: dict {block_id: coeff}
    # targets: list of (block_id, coeff)
    strassen_terms = [
        ({11: 1, 22: 1}, {11: 1, 22: 1}, [(11, 1), (22, 1)]), # M1
        ({21: 1, 22: 1}, {11: 1}, [(21, 1), (22, -1)]),       # M2
        ({11: 1}, {12: 1, 22: -1}, [(12, 1), (22, 1)]),       # M3
        ({22: 1}, {21: 1, 11: -1}, [(11, 1), (21, 1)]),       # M4
        ({11: 1, 12: 1}, {22: 1}, [(11, -1), (12, 1)]),       # M5
        ({21: 1, 11: -1}, {11: 1, 12: 1}, [(22, 1)]),         # M6
        ({12: 1, 22: -1}, {21: 1, 22: 1}, [(11, 1)])          # M7
    ]
    
    col = 0
    for l in range(7):
        if time.time() > deadline:
            break
        X_blocks, Y_blocks, targets = strassen_terms[l]
        for u in range(3):
            for v in range(3):
                for w in range(3):
                    # U and V contributions
                    for b_x, c_x in X_blocks.items():
                        row_off, col_off = offsets[b_x]
                        i = u + row_off
                        j = v + col_off
                        U[i * m + j][col] += c_x
                        
                    for b_y, c_y in Y_blocks.items():
                        row_off, col_off = offsets[b_y]
                        j = v + row_off
                        k = w + col_off
                        V[j * p + k][col] += c_y
                        
                    # W contributions
                    for b_t, c_t in targets:
                        row_off, col_off = offsets[b_t]
                        i_c = u + row_off
                        k_c = w + col_off
                        W[k_c * n + i_c][col] += c_t
                        
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
