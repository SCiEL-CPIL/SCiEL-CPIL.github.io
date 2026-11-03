import numpy as np

def construct():
    n, m, p = 6, 6, 6
    rank_outer = 27
    rank_inner = 7
    total_rank = rank_outer * rank_inner
    
    # Initialize U, V, W as lists of lists of zeros
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 (Rank 7)
    # Indices: 00->0, 01->1, 10->2, 11->3
    U_in = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 0, 1, 0], # P5
        [-1, 0, 1, 0],# P6
        [0, 1, 0, -1] # P7
    ]
    V_in = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 0, 1, -1],# P3
        [-1, 1, 0, 0],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    W_in = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, -1],# P2
        [0, 1, 0, 1], # P3
        [1, 0, 1, 0], # P4
        [-1, 1, 0, 0],# P5
        [0, 0, 0, 1], # P6
        [-1, 0, 0, 0] # P7
    ]
    
    # Schoolbook decomposition for 3x3 (Rank 27)
    # U_out[i1, j1, r], V_out[j1, k1, r], W_out[k1, i1, r]
    # r = i1*3 + j1? No, r is the term index.
    # For schoolbook, term r corresponds to (i1, j1) pair? No.
    # Term r corresponds to (i1, j1, k1) triple? No.
    # Schoolbook: C[i1, k1] = sum_j1 A[i1, j1] B[j1, k1].
    # Each term is A[i1, j1] * B[j1, k1] * C[i1, k1].
    # There are 3*3*3 = 27 terms.
    U_out = [[0]*9 for _ in range(27)]
    V_out = [[0]*9 for _ in range(27)]
    W_out = [[0]*9 for _ in range(27)]
    
    r = 0
    for i1 in range(3):
        for j1 in range(3):
            for k1 in range(3):
                # A index: i1, j1 -> i1*3 + j1
                U_out[r][i1 * 3 + j1] = 1
                # B index: j1, k1 -> j1*3 + k1
                V_out[r][j1 * 3 + k1] = 1
                # C index: k1, i1? No, W is for C[i1, k1].
                # In the tensor T, W is indexed by (row_C, col_C).
                # row_C = i1, col_C = k1.
                # So W_out[r][i1 * 3 + k1] = 1.
                W_out[r][i1 * 3 + k1] = 1
                r += 1
                
    # Fill U, V, W
    for i1 in range(3):
        for i2 in range(2):
            for j1 in range(3):
                for j2 in range(2):
                    I = 2 * i1 + i2
                    J = 2 * j1 + j2
                    row_U = I * 6 + J
                    for r in range(27):
                        u_val = U_out[r][i1 * 3 + j1]
                        if u_val == 0: continue
                        for s in range(7):
                            u_in_val = U_in[s][i2 * 2 + j2]
                            if u_in_val == 0: continue
                            col = r * 7 + s
                            U[row_U][col] = u_val * u_in_val
                            
    for j1 in range(3):
        for j2 in range(2):
            for k1 in range(3):
                for k2 in range(2):
                    J = 2 * j1 + j2
                    K = 2 * k1 + k2
                    row_V = J * 6 + K
                    for r in range(27):
                        v_val = V_out[r][j1 * 3 + k1]
                        if v_val == 0: continue
                        for s in range(7):
                            v_in_val = V_in[s][j2 * 2 + k2]
                            if v_in_val == 0: continue
                            col = r * 7 + s
                            V[row_V][col] = v_val * v_in_val
                            
    for k1 in range(3):
        for k2 in range(2):
            for i1 in range(3):
                for i2 in range(2):
                    K = 2 * k1 + k2
                    I = 2 * i1 + i2
                    row_W = K * 6 + I
                    for r in range(27):
                        w_val = W_out[r][i1 * 3 + k1]
                        if w_val == 0: continue
                        for s in range(7):
                            w_in_val = W_in[s][k2 * 2 + i2]
                            if w_in_val == 0: continue
                            col = r * 7 + s
                            W[row_W][col] = w_val * w_in_val

    return {"U": U, "V": V, "W": W}
