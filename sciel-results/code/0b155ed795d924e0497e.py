"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Rank 42 decomposition for M_{2,4,6}
    # Split A (2x4) into A1, A2 (2x2 each)
    # Split B (4x6) into B1, B2 (2x6 each), then each Bi into three 2x2 blocks
    # Total 2 * 3 * rank(M_{2,2,2}) = 2 * 3 * 7 = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for M_{2,2,2}
    # U_coeffs[s][(i, j)] = coeff
    # V_coeffs[s][(j, k)] = coeff
    # W_coeffs[s][(k, i)] = coeff
    
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},      # M1
        {(1, 0): 1, (1, 1): 1},      # M2
        {(0, 0): 1},                 # M3
        {(1, 1): 1},                 # M4
        {(0, 0): 1, (0, 1): 1},      # M5
        {(1, 0): 1, (0, 0): -1},     # M6
        {(0, 1): 1, (1, 1): -1}      # M7
    ]
    
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},      # M1
        {(0, 0): 1},                 # M2
        {(0, 1): 1, (1, 1): -1},     # M3
        {(1, 0): 1, (0, 0): -1},     # M4
        {(1, 1): 1},                 # M5
        {(0, 0): 1, (0, 1): 1},      # M6
        {(1, 0): 1, (1, 1): 1}       # M7
    ]
    
    W_coeffs = [
        {(0, 0): 1, (1, 1): 1},      # M1
        {(1, 0): 1, (1, 1): -1},     # M2
        {(0, 1): 1, (1, 1): 1},      # M3
        {(1, 0): 1, (0, 0): 1},      # M4
        {(0, 1): 1, (0, 0): -1},     # M5
        {(1, 1): 1},                 # M6
        {(0, 0): 1}                  # M7
    ]
    
    col = 0
    for a_block in range(2):  # A blocks (j major block)
        for b_block in range(3):  # B column blocks (k major block)
            for s in range(7):  # Strassen terms
                # U part: A[i, j_glob]
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a_block * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # V part: B[j_glob, k_glob]
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a_block * 2 + j_loc
                    k = b_block * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # W part: C[k_glob, i]
                for (k_loc, i_loc), coeff in W_coeffs[s].items():
                    i = i_loc
                    k = b_block * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
