"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # 140s safety margin

def construct():
    n, m, p = 4, 4, 5
    rank_total = 49 + 16  # 65
    
    U = [[0] * rank_total for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank_total for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank_total for _ in range(p * n)]  # 20 x 65
    
    # Part 2: 4x4x1 (B column 4), rank 16
    col = 49
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    # Part 1: 4x4x4, rank 49 (Strassen)
    strassen_2x2 = [
        ((1,0,0,1), (1,0,0,1), (1,0,0,0)),
        ((0,0,1,1), (1,0,0,0), (0,0,1,0)),
        ((1,0,0,0), (0,1,-1,0), (0,1,0,0)),
        ((0,0,0,1), (-1,0,1,0), (1,0,0,0)),
        ((1,1,0,0), (0,0,0,1), (0,1,0,0)),
        ((-1,0,1,0), (1,1,0,0), (0,0,0,1)),
        ((0,1,-1,0), (0,1,1,0), (1,0,0,1)),
    ]
    
    super_product_defs = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    block_offsets = {0: (0,0), 1: (0,2), 2: (2,0), 3: (2,2)}
    
    col = 0
    for sp_idx, (x_coeffs, y_coeffs, c_contrib) in enumerate(super_product_defs):
        for t_idx, (p_coeff, q_coeff, r_coeff) in enumerate(strassen_2x2):
            for block, coeff in x_coeffs.items():
                bro, bco = block_offsets[block]
                for r in range(2):
                    for c in range(2):
                        pc = p_coeff[r*2+c]
                        if pc != 0:
                            a_row = bro + r
                            a_col = bco + c
                            U[a_row * m + a_col][col] += coeff * pc
            
            for block, coeff in y_coeffs.items():
                bro, bco = block_offsets[block]
                for r in range(2):
                    for c in range(2):
                        qc = q_coeff[r*2+c]
                        if qc != 0:
                            b_row = bro + r
                            b_col = bco + c
                            V[b_row * p + b_col][col] += coeff * qc
            
            for cblock, coeff in c_contrib.items():
                bro, bco = block_offsets[cblock]
                for r in range(2):
                    for c in range(2):
                        rc = r_coeff[r*2+c]
                        if rc != 0:
                            c_row = bro + r
                            c_col = bco + c
                            W[c_col * n + c_row][col] += coeff * rc
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
