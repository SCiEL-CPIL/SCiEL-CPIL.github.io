import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # Each term is a tuple (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: list of (i, j, val) for A block
    # b_coeffs: list of (j, k, val) for B block
    # c_coeffs: list of (i, k, val) for C block (output)
    
    strassen_terms = [
        # P1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # P2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # P3
        ([(0,0,1)], [(1,0,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # P4
        ([(1,1,1)], [(0,1,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # P5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # P6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)], [(1,1,1)]),
        # P7
        ([(0,1,1), (1,1,-1)], [(1,1,1), (0,1,1)], [(0,0,1), (1,1,-1)])
    ]
    
    # Blocks:
    # Block 0: A_L (cols 0-1), B_LL (rows 0-1, cols 0-1)
    # Block 1: A_L (cols 0-1), B_LR (rows 0-1, cols 2-3)
    # Block 2: A_R (cols 2-3), B_RL (rows 2-3, cols 0-1)
    # Block 3: A_R (cols 2-3), B_RR (rows 2-3, cols 2-3)
    
    # j_offset for A, k_offset for B
    # A is 2x4. j is 0..3.
    # B is 4x4. j is 0..3, k is 0..3.
    
    block_configs = [
        # (j_offset_A, j_offset_B, k_offset_B)
        (0, 0, 0), # Block 0
        (0, 0, 2), # Block 1
        (2, 2, 0), # Block 2
        (2, 2, 2)  # Block 3
    ]
    
    col = 0
    for j_off_A, j_off_B, k_off_B in block_configs:
        for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
            # Update U
            for i, j, val in a_coeffs:
                # Global index in U: i * m + (j_off_A + j)
                u_idx = i * m + (j_off_A + j)
                U[u_idx][col] += val
            
            # Update V
            for j, k, val in b_coeffs:
                # Global index in V: (j_off_B + j) * p + (k_off_B + k)
                v_idx = (j_off_B + j) * p + (k_off_B + k)
                V[v_idx][col] += val
            
            # Update W
            for i, k, val in c_coeffs:
                # Global index in W: (k_off_B + k) * n + i
                w_idx = (k_off_B + k) * n + i
                W[w_idx][col] += val
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
