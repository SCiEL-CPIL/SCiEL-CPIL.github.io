import numpy as np

DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS
    # p = 14
    # We split the 14 columns into 7 blocks of 2 columns each.
    # For each block, we use the rank 7 Strassen decomposition for 2x2x2.
    # Total rank = 7 * 7 = 49.
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)] # 4 x 49
    V = [[0] * rank for _ in range(m * p)] # 28 x 49
    W = [[0] * rank for _ in range(p * n)] # 28 x 49
    
    # Strassen decomposition for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)h
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)g
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(f+h)
    # C = [[P1-P4-P5+P7, P3+P5], [P2+P4, P1-P2+P3+P6]]
    
    # Indices for A (2x2):
    # a=0, b=1, c=2, d=3
    # Indices for B block (2x2):
    # e, f, g, h
    
    # Coefficients for U (A part)
    # P1: a+d -> [1, 0, 0, 1]
    # P2: c+d -> [0, 0, 1, 1]
    # P3: a   -> [1, 0, 0, 0]
    # P4: d   -> [0, 0, 0, 1]
    # P5: a+b -> [1, 1, 0, 0]
    # P6: c-a -> [-1, 0, 1, 0]
    # P7: b-d -> [0, 1, 0, -1]
    
    U_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # Coefficients for V (B part) for a single 2x2 block
    # P1: e+h -> [1, 0, 0, 1] (assuming B block is [[e, f], [g, h]])
    # P2: h   -> [0, 0, 0, 1]
    # P3: f-h -> [0, 1, 0, -1]
    # P4: g-e -> [-1, 0, 1, 0]
    # P5: g   -> [0, 0, 1, 0]
    # P6: e+f -> [1, 1, 0, 0]
    # P7: f+h -> [0, 1, 0, 1]
    
    V_coeffs = [
        [1, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 1]
    ]
    
    # Coefficients for W (C part)
    # C00 = P1 - P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # W is indexed by [k*n + i] where k is col index in B, i is row index in C.
    # For a block at columns 2*t and 2*t+1:
    # C[0, 2t] corresponds to W index (2t)*2 + 0 = 4t
    # C[0, 2t+1] corresponds to W index (2t+1)*2 + 0 = 4t+2
    # C[1, 2t] corresponds to W index (2t)*2 + 1 = 4t+1
    # C[1, 2t+1] corresponds to W index (2t+1)*2 + 1 = 4t+3
    
    W_coeffs = [
        # P1
        [1, 0, 0, 1], # C00, C01, C10, C11
        # P2
        [0, 0, 1, -1],
        # P3
        [0, 1, 0, 1],
        # P4
        [-1, 0, 1, 0],
        # P5
        [-1, 1, 0, 0],
        # P6
        [0, 0, 0, 1],
        # P7
        [1, 0, 0, 0]
    ]
    
    # Note: The W_coeffs above map P_r to the 4 outputs of the 2x2 block.
    # We need to distribute these into the global W matrix.
    
    for t in range(7):
        base_col = 2 * t
        for r in range(7):
            col = t * 7 + r
            # Fill U
            for i in range(4):
                U[i][col] = U_coeffs[r][i]
            # Fill V
            # V index is j*p + k.
            # j=0, k=base_col -> 0*14 + base_col = base_col
            # j=0, k=base_col+1 -> base_col+1
            # j=1, k=base_col -> 1*14 + base_col = 14 + base_col
            # j=1, k=base_col+1 -> 14 + base_col + 1
            v_indices = [base_col, base_col + 1, 14 + base_col, 14 + base_col + 1]
            for idx, v_idx in enumerate(v_indices):
                V[v_idx][col] = V_coeffs[r][idx]
            # Fill W
            # W index is k*n + i.
            # k=base_col, i=0 -> base_col*2 + 0
            # k=base_col+1, i=0 -> (base_col+1)*2 + 0
            # k=base_col, i=1 -> base_col*2 + 1
            # k=base_col+1, i=1 -> (base_col+1)*2 + 1
            w_indices = [2*base_col, 2*base_col+2, 2*base_col+1, 2*base_col+3]
            # Wait, W index is k*n + i.
            # C[i, k].
            # C[0, base_col] -> W[base_col*2 + 0]
            # C[0, base_col+1] -> W[(base_col+1)*2 + 0]
            # C[1, base_col] -> W[base_col*2 + 1]
            # C[1, base_col+1] -> W[(base_col+1)*2 + 1]
            
            # My W_coeffs order was C00, C01, C10, C11.
            # So:
            W[w_indices[0]][col] = W_coeffs[r][0]
            W[w_indices[1]][col] = W_coeffs[r][1]
            W[w_indices[2]][col] = W_coeffs[r][2]
            W[w_indices[3]][col] = W_coeffs[r][3]
            
    return {"U": U, "V": V, "W": W}
