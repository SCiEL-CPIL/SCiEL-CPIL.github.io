import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 49 + 49 = 98
    rank = 98
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen steps for 2x2
    # (L_indices, R_indices, W_indices)
    # Indices are 0:00, 1:01, 2:10, 3:11
    S2 = [
        ([0, 3], [0, 3], [0, 3]),
        ([2, 3], [0],    [2, 3]),
        ([0],    [1, 3], [1, 3]),
        ([3],    [2, 0], [0, 2]),
        ([0, 1], [3],    [0, 1]),
        ([2, 0], [0, 1], [3]),
        ([1, 3], [2, 3], [0]),
    ]
    # Coefficients for L, R, W
    # We need to handle signs.
    # S1: L=1,1; R=1,1; W=1,1
    # S2: L=1,1; R=1; W=1,-1
    # S3: L=1; R=1,-1; W=1,1
    # S4: L=1; R=-1,1; W=1,1 (Wait, S4 is X11*(Y10-Y00). Y10 is idx 2, Y00 is idx 0. So R is [2,0] with coeffs [1,-1]. W is Z00, Z10. Z00 is 0, Z10 is 2. Coeffs 1,1.)
    # S5: L=1,1; R=1; W=-1,1
    # S6: L=-1,1; R=1,1; W=1
    # S7: L=1,-1; R=1,1; W=1
    
    S2_coeffs = [
        ([(0,1), (3,1)], [(0,1), (3,1)], [(0,1), (3,1)]),
        ([(2,1), (3,1)], [(0,1)],         [(2,1), (3,-1)]),
        ([(0,1)],         [(1,1), (3,-1)],[(1,1), (3,1)]),
        ([(3,1)],         [(2,1), (0,-1)],[(0,1), (2,1)]),
        ([(0,1), (1,1)],  [(3,1)],        [(0,-1), (1,1)]),
        ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]),
        ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1)]),
    ]
    
    # Strassen steps for 4x4 (blocks)
    # Block indices: 0:00, 1:01, 2:10, 3:11
    S4 = [
        ([(0,1), (3,1)], [(0,1), (3,1)]), # M1
        ([(2,1), (3,1)], [(0,1)]),         # M2
        ([(0,1)],         [(1,1), (3,-1)]),# M3
        ([(3,1)],         [(2,1), (0,-1)]),# M4
        ([(0,1), (1,1)],  [(3,1)]),        # M5
        ([(2,1), (0,-1)], [(0,1), (1,1)]), # M6
        ([(1,1), (3,-1)], [(2,1), (3,1)]), # M7
    ]
    
    # C contributions for each M_k
    # C blocks: 0:00, 1:01, 2:10, 3:11
    C_contribs = [
        [(0,1), (3,1)], # M1
        [(2,1), (3,-1)],# M2
        [(1,1), (3,1)], # M3
        [(0,1), (2,1)], # M4
        [(0,-1), (1,1)],# M5
        [(3,1)],        # M6
        [(0,1)],        # M7
    ]
    
    for half in range(2):
        b_offset = half * 4
        c_offset = half * 4
        col_base = half * 49
        
        for k in range(7):
            L_blocks = S4[k][0]
            R_blocks = S4[k][1]
            
            for sub_k in range(7):
                L_sub = S2_coeffs[sub_k][0]
                R_sub = S2_coeffs[sub_k][1]
                W_sub = S2_coeffs[sub_k][2]
                
                col_idx = col_base + k * 7 + sub_k
                
                # U
                for idx, coeff_sub in L_sub:
                    u_row_in_comb = idx // 2
                    u_col_in_comb = idx % 2
                    for b_idx, coeff_block in L_blocks:
                        bi = b_idx // 2
                        bj = b_idx % 2
                        a_row = 2 * bi + u_row_in_comb
                        a_col = 2 * bj + u_col_in_comb
                        u_idx = a_row * 4 + a_col
                        U[u_idx, col_idx] += coeff_block * coeff_sub
                        
                # V
                for idx, coeff_sub in R_sub:
                    v_row_in_comb = idx // 2
                    v_col_in_comb = idx % 2
                    for b_idx, coeff_block in R_blocks:
                        bi = b_idx // 2
                        bj = b_idx % 2
                        b_row = 2 * bi + v_row_in_comb
                        b_col = 2 * bj + v_col_in_comb + b_offset
                        v_idx = b_row * 8 + b_col
                        V[v_idx, col_idx] += coeff_block * coeff_sub
                        
                # W
                for idx, coeff_sub in W_sub:
                    w_row_in_comb = idx // 2
                    w_col_in_comb = idx % 2
                    for c_idx, coeff_c in C_contribs[k]:
                        ci = c_idx // 2
                        cj = c_idx % 2
                        c_row = 2 * ci + w_row_in_comb
                        c_col = 2 * cj + w_col_in_comb + c_offset
                        w_idx = c_col * 4 + c_row
                        W[w_idx, col_idx] += coeff_c * coeff_sub
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
