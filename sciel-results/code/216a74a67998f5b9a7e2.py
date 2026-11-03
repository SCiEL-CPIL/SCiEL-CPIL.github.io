import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen on 2x2x2 blocks
    # Structure:
    # 2x4x8 = 2 * (2x2x8)
    # 2x2x8 = 2 * (2x2x4)
    # 2x2x4 = 2 * (2x2x2)
    # 2x2x2 = 7 (Strassen)
    # Total: 2 * 2 * 2 * 7 = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definitions
    # Each term is a tuple (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict mapping (i,j) to coeff
    # B_coeffs: dict mapping (i,j) to coeff
    # C_coeffs: dict mapping (i,j) to coeff
    # Local indices 0,1 for 2x2
    
    strassen_terms = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): -1}),
    ]
    
    col = 0
    # Iterate over the 8 blocks of size 2x2x2
    # j_start in {0, 2}
    # k_start in {0, 2, 4, 6}
    for j_start in [0, 2]:
        for k_start in [0, 2, 4, 6]:
            for A_c, B_c, C_c in strassen_terms:
                # Update U (A part)
                for (i_loc, j_loc), coeff in A_c.items():
                    i_glob = i_loc
                    j_glob = j_start + j_loc
                    row_u = i_glob * m + j_glob
                    U[row_u][col] += coeff
                    
                # Update V (B part)
                for (j_loc, k_loc), coeff in B_c.items():
                    j_glob = j_start + j_loc
                    k_glob = k_start + k_loc
                    row_v = j_glob * p + k_glob
                    V[row_v][col] += coeff
                    
                # Update W (C part)
                for (k_loc, i_loc), coeff in C_c.items():
                    k_glob = k_start + k_loc
                    i_glob = i_loc
                    row_w = k_glob * n + i_glob
                    W[row_w][col] += coeff
                    
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
