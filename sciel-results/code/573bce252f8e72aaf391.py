"""Tensor decomposition for 2x4 * 4x6 matrix multiplication using block Strassen."""

def construct():
    n, m, p = 2, 4, 6
    # We partition the 4x6 matrix into three 2x2 blocks and the 2x4 matrix into two 2x2 blocks.
    # Each 2x2 * 2x2 multiplication uses Strassen's 7-term algorithm.
    # Total rank = 2 * 3 * 7 = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 7-term algorithm for 2x2 * 2x2:
    # M = [[m00, m01], [m10, m11]], N = [[n00, n01], [n10, n11]]
    # P1 = (m00 + m11) * (n00 + n11)
    # P2 = (m10 + m11) * n00
    # P3 = m00 * (n10 - n00)
    # P4 = m11 * (n01 - n11)
    # P5 = (m00 + m01) * n11
    # P6 = (m10 - m00) * (n00 + n01)
    # P7 = (m01 - m11) * (n10 + n11)
    # R00 = P1 + P4 - P5 + P7
    # R01 = P3 + P5
    # R10 = P2 + P4
    # R11 = P1 - P2 + P3 + P6
    
    # Define Strassen coefficients for U (first matrix M):
    # U_coeffs[s] = dict mapping (i, j) to coefficient for the s-th Strassen product
    U_s = [
        {(0, 0): 1, (1, 1): 1},      # P1: m00 + m11
        {(1, 0): 1, (1, 1): 1},      # P2: m10 + m11
        {(0, 0): 1},                  # P3: m00
        {(1, 1): 1},                  # P4: m11
        {(0, 0): 1, (0, 1): 1},      # P5: m00 + m01
        {(1, 0): 1, (0, 0): -1},     # P6: m10 - m00
        {(0, 1): 1, (1, 1): -1}      # P7: m01 - m11
    ]
    
    # Define Strassen coefficients for V (second matrix N):
    # V_coeffs[s] = dict mapping (j, k) to coefficient for the s-th Strassen product
    V_s = [
        {(0, 0): 1, (1, 1): 1},      # P1: n00 + n11
        {(0, 0): 1},                  # P2: n00
        {(1, 0): 1, (0, 0): -1},     # P3: n10 - n00
        {(0, 1): 1, (1, 1): -1},     # P4: n01 - n11
        {(1, 1): 1},                  # P5: n11
        {(0, 0): 1, (0, 1): 1},      # P6: n00 + n01
        {(1, 0): 1, (1, 1): 1}       # P7: n10 + n11
    ]
    
    # Define Strassen coefficients for W (output R):
    # W_coeffs[s] = dict mapping (i, k) to coefficient for the s-th Strassen product
    W_s = [
        {(0, 0): 1, (1, 1): 1},      # P1: contributes to R00 (+1) and R11 (+1)
        {(1, 0): 1, (1, 1): -1},     # P2: contributes to R10 (+1) and R11 (-1)
        {(0, 1): 1, (1, 1): 1},      # P3: contributes to R01 (+1) and R11 (+1)
        {(0, 0): 1, (1, 0): 1},      # P4: contributes to R00 (+1) and R10 (+1)
        {(0, 0): -1, (0, 1): 1},     # P5: contributes to R00 (-1) and R01 (+1)
        {(1, 1): 1},                 # P6: contributes to R11 (+1)
        {(0, 0): 1}                  # P7: contributes to R00 (+1)
    ]
    
    col = 0
    for a in range(2):  # A block index: a=0 for cols 0-1, a=1 for cols 2-3
        for b in range(3):  # B col block index: b=0 for cols 0-1, b=1 for 2-3, b=2 for 4-5
            for s in range(7):  # Strassen term
                # Update U
                for (i_loc, j_loc), coeff in U_s[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # Update V
                for (j_loc, k_loc), coeff in V_s[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # Update W
                for (i_loc, k_loc), coeff in W_s[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
