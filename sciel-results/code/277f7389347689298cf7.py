"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition of 6x6 into 3x3 blocks of 2x2 matrices.
    # Each 2x2 block multiplication uses Strassen's algorithm (rank 7).
    # Total rank = 3 * 3 * 3 * 7 = 189.
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen vectors for 2x2 multiplication
    # Order: [00, 01, 10, 11]
    U_s = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    V_s = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W_s = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for s in range(7):
                    # Map local 2x2 indices to global 6x6 indices
                    # A block (i, j) -> rows 2i, 2i+1; cols 2j, 2j+1
                    # B block (j, k) -> rows 2j, 2j+1; cols 2k, 2k+1
                    # C block (i, k) -> rows 2i, 2i+1; cols 2k, 2k+1
                    
                    for a in range(2):
                        for b in range(2):
                            val_u = U_s[s][a * 2 + b]
                            val_v = V_s[s][a * 2 + b]
                            val_w = W_s[s][a * 2 + b]
                            
                            if val_u != 0:
                                row_u = (2 * i + a) * m + (2 * j + b)
                                U[row_u][col] = val_u
                                
                            if val_v != 0:
                                row_v = (2 * j + a) * p + (2 * k + b)
                                V[row_v][col] = val_v
                                
                            if val_w != 0:
                                # W index convention: W[k*n + i, r]
                                # Global C row is 2*i + a, col is 2*k + b
                                # So i_global = 2*i + a, k_global = 2*k + b
                                row_w = (2 * k + b) * n + (2 * i + a)
                                W[row_w][col] = val_w
                    col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
