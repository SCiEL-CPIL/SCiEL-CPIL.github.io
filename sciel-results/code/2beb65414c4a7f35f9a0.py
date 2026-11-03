import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen terms for 2x2x2
    # u, v, w are vectors of length 4 (flattened 2x2 matrices)
    # Indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    strassen_terms = [
        # u, v, w
        ([0, 0, -1, 1], [0, 0, 1, 1], [1, 0, 0, 1]),   # m1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),     # m2
        ([1, 0, 0, 0], [1, 0, 0, -1], [0, 1, 0, 0]),    # m3
        ([0, 0, 0, 1], [-1, 0, 0, 1], [-1, 0, 0, 1]),   # m4
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 1, 0]),     # m5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 1]),    # m6
        ([0, 1, 0, 1], [0, 0, 1, -1], [1, 0, 0, 0])     # m7
    ]
    
    # Block coefficients for Strassen steps
    # x[k][alpha] is coeff of A_alpha in X_k
    # y[k][beta] is coeff of B_beta in Y_k
    # out_blocks[k] is dict {block_idx: coeff}
    # Block indices: 00->0, 01->1, 10->2, 11->3
    block_x = [
        [0, 0, -1, 1], # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 0, 0, 1],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, 1]   # M7
    ]
    block_y = [
        [0, 0, 1, 1], # M1
        [1, 0, 0, 0], # M2
        [1, 0, 0, -1],# M3
        [-1, 0, 0, 1],# M4
        [1, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, -1] # M7
    ]
    block_out = [
        {0: 1, 3: 1},  # M1 -> C00, C11
        {2: 1},        # M2 -> C10
        {1: 1},        # M3 -> C01
        {0: -1, 3: 1}, # M4 -> C00, C11
        {0: 1, 2: 1},  # M5 -> C00, C10
        {1: 1, 3: 1},  # M6 -> C01, C11
        {0: 1}         # M7 -> C00
    ]
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # First 49 terms for 4x4x4 part
    for k in range(7):
        for s in range(7):
            u_s, v_s, w_s = strassen_terms[s]
            x_k = block_x[k]
            y_k = block_y[k]
            out_blocks = block_out[k]
            
            # u_s, v_s, w_s are for 2x2 matrices
            # We need to expand these to 4x4 matrices
            # A_alpha is a 2x2 block. A_flat index for A_alpha[i,j] is (alpha_row*2 + i)*4 + (alpha_col*2 + j)
            # B_beta is a 2x2 block. B_flat index for B_beta[i,j] is (beta_row*2 + i)*5 + (beta_col*2 + j)
            # C_gamma is a 2x2 block. C_flat index for C_gamma[i,j] is (gamma_col*2 + j)*4 + (gamma_row*2 + i)
            
            for alpha in range(4):
                if x_k[alpha] == 0: continue
                ar = alpha // 2
                ac = alpha % 2
                for i_loc in range(2):
                    for j_loc in range(2):
                        coeff_u = x_k[alpha] * u_s[i_loc * 2 + j_loc]
                        if coeff_u == 0: continue
                        i_global = ar * 2 + i_loc
                        j_global = ac * 2 + j_loc
                        U[i_global * m + j_global][col] += coeff_u
                        
            for beta in range(4):
                if y_k[beta] == 0: continue
                br = beta // 2
                bc = beta % 2
                for i_loc in range(2):
                    for j_loc in range(2):
                        coeff_v = y_k[beta] * v_s[i_loc * 2 + j_loc]
                        if coeff_v == 0: continue
                        j_global = br * 2 + i_loc # Row of B
                        k_global = bc * 2 + j_loc # Col of B
                        V[j_global * p + k_global][col] += coeff_v
                        
            for gamma, coeff_w_block in out_blocks.items():
                gr = gamma // 2
                gc = gamma % 2
                for i_loc in range(2):
                    for j_loc in range(2):
                        coeff_w = coeff_w_block * w_s[i_loc * 2 + j_loc]
                        if coeff_w == 0: continue
                        i_global = gr * 2 + i_loc
                        k_global = gc * 2 + j_loc
                        W[k_global * n + i_global][col] += coeff_w
                        
            col += 1
            
    # Last 16 terms for 4x4x1 part (column 4)
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
