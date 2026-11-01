"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use 5 Strassen blocks for the 10 columns (2 cols per block)
    # Rank = 5 * 7 = 35
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 Strassen steps for a 2x2 * 2x2 multiplication
    # Each step is a tuple: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {row_idx: coeff}
    # V_coeffs: dict {row_idx: coeff}
    # W_coeffs: dict {row_idx: coeff}
    
    # Local indices for A (U):
    # 0: a00, 1: a01, 2: a10, 3: a11
    
    # Local indices for B (V) relative to block start:
    # 0: b00, 1: b01, 2: b10, 3: b11
    # Note: Global V row = j * p + k
    # For block b, k is 2*b or 2*b+1
    # j=0 -> k=2b, 2b+1
    # j=1 -> k=2b, 2b+1
    
    # Local indices for C (W) relative to block start:
    # 0: c00, 1: c01, 2: c10, 3: c11
    # Note: Global W row = k * n + i
    # For block b, k is 2*b or 2*b+1
    # k=2b -> i=0,1 -> rows 4b, 4b+1
    # k=2b+1 -> i=0,1 -> rows 4b+2, 4b+3
    
    strassen_steps = [
        {
            "U": {0: 1, 3: 1},       # (a00 + a11)
            "V": {0: 1, 2: 1},       # (b00 + b10)
            "W": {0: 1, 3: 1}        # c00 += 1, c11 += 1
        },
        {
            "U": {2: 1, 3: 1},       # (a10 + a11)
            "V": {0: 1},             # b00
            "W": {2: 1, 3: -1}       # c10 += 1, c11 -= 1
        },
        {
            "U": {0: 1},             # a00
            "V": {1: 1, 3: -1},      # (b01 - b11)
            "W": {1: 1, 3: 1}        # c01 += 1, c11 += 1
        },
        {
            "U": {3: 1},             # a11
            "V": {2: 1, 0: -1},      # (b10 - b00)
            "W": {0: 1, 2: 1}        # c00 += 1, c10 += 1
        },
        {
            "U": {0: 1, 1: 1},       # (a00 + a01)
            "V": {3: 1},             # b11
            "W": {0: -1, 1: 1}       # c00 -= 1, c01 += 1
        },
        {
            "U": {2: 1, 0: -1},      # (a10 - a00)
            "V": {0: 1, 1: 1},       # (b00 + b01)
            "W": {3: 1}              # c11 += 1
        },
        {
            "U": {1: 1, 3: -1},      # (a01 - a11)
            "V": {2: 1, 3: 1},       # (b10 + b11)
            "W": {0: 1}              # c00 += 1
        }
    ]
    
    num_blocks = p // 2  # 5 blocks
    
    for b in range(num_blocks):
        # Offset for V and W indices
        # V: row = j * p + k. k = 2*b + local_k
        # W: row = k * n + i. k = 2*b + local_k. row = (2*b + local_k)*2 + local_i
        
        for s in range(7):
            col_idx = b * 7 + s
            step = strassen_steps[s]
            
            # Fill U
            for u_row, coeff in step["U"].items():
                U[u_row][col_idx] += coeff
                
            # Fill V
            for v_local_row, coeff in step["V"].items():
                # v_local_row 0->b00, 1->b01, 2->b10, 3->b11
                # v_local_row // 2 is j (0 or 1)
                # v_local_row % 2 is offset in k (0 or 1)
                j = v_local_row // 2
                k_offset = v_local_row % 2
                k = 2 * b + k_offset
                v_row = j * p + k
                V[v_row][col_idx] += coeff
                
            # Fill W
            for w_local_row, coeff in step["W"].items():
                # w_local_row 0->c00, 1->c01, 2->c10, 3->c11
                # w_local_row // 2 is k_offset (0 or 1)
                # w_local_row % 2 is i (0 or 1)
                k_offset = w_local_row // 2
                i = w_local_row % 2
                k = 2 * b + k_offset
                w_row = k * n + i
                W[w_row][col_idx] += coeff
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
