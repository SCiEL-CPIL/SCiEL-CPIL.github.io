import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 120.0

    n, m, p = DIMENSIONS  # 6, 6, 6
    # Block size for Strassen decomposition
    s = 3  # 2x2 blocks of size 3x3
    
    # Strassen definitions for 2x2 block multiplication
    # Each entry is a list of (block_row, block_col, coeff)
    # X_k: Left matrix factors for M_k
    # Y_k: Right matrix factors for M_k
    # C_k: Output block contributions for M_k (block_row, block_col, coeff)
    
    strassen_X = [
        [(0, 0, 1), (1, 1, 1)],  # M1: A00 + A11
        [(1, 0, 1), (1, 1, 1)],  # M2: A10 + A11
        [(0, 0, 1)],             # M3: A00
        [(1, 1, 1)],             # M4: A11
        [(0, 0, 1), (0, 1, 1)],  # M5: A00 + A01
        [(1, 0, 1), (0, 0, -1)], # M6: A10 - A00
        [(0, 1, 1), (1, 1, -1)]  # M7: A01 - A11
    ]
    
    strassen_Y = [
        [(0, 0, 1), (1, 1, 1)],  # M1: B00 + B11
        [(0, 0, 1)],             # M2: B00
        [(0, 1, 1), (1, 1, -1)], # M3: B01 - B11
        [(1, 0, 1), (0, 0, -1)], # M4: B10 - B00
        [(1, 1, 1)],             # M5: B11
        [(0, 0, 1), (0, 1, 1)],  # M6: B00 + B01
        [(1, 0, 1), (1, 1, 1)]   # M7: B10 + B11
    ]
    
    # Output blocks C_pr = sum_k e_k M_k
    # e_k for each output block (pr)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    strassen_C = [
        [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)], # M1: C00(1), C01(1? No), ... 
        # Wait, let's map k to (P_out, R_out, coeff)
        # M1 contributes to C00 (1) and C11 (1)
        [(0, 0, 1), (1, 1, 1)],
        # M2 contributes to C10 (1) and C11 (-1)
        [(1, 0, 1), (1, 1, -1)],
        # M3 contributes to C01 (1) and C11 (1)
        [(0, 1, 1), (1, 1, 1)],
        # M4 contributes to C00 (1) and C10 (1)
        [(0, 0, 1), (1, 0, 1)],
        # M5 contributes to C00 (-1) and C01 (1)
        [(0, 0, -1), (0, 1, 1)],
        # M6 contributes to C11 (1)
        [(1, 1, 1)],
        # M7 contributes to C00 (1)
        [(0, 0, 1)]
    ]

    rank = 7 * s * s * s  # 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    rank_idx = 0
    for k in range(7):
        X_terms = strassen_X[k]
        Y_terms = strassen_Y[k]
        C_terms = strassen_C[k]
        
        # Iterate over local indices within the 3x3 blocks
        for u in range(s):
            for v in range(s):
                for w in range(s):
                    # This (k, u, v, w) corresponds to one rank-1 term
                    # U: A block (P, Q) at local (u, v)
                    # V: B block (Q', R) at local (v, w)
                    # W: C block (P_out, R_out) at local (u, w)
                    
                    for (P, Q, c_x) in X_terms:
                        i = P * s + u
                        j = Q * s + v
                        U[i * n + j][rank_idx] += c_x

                    for (Q_prime, R, c_y) in Y_terms:
                        j = Q_prime * s + v
                        k_idx = R * s + w
                        V[j * m + k_idx][rank_idx] += c_y

                    for (P_out, R_out, c_w) in C_terms:
                        k_idx = R_out * s + w
                        i = P_out * s + u
                        W[k_idx * p + i][rank_idx] += c_w

                    rank_idx += 1
                    if time.time() > deadline - 5.0:
                        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
