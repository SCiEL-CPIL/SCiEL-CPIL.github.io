import numpy as np

def construct():
    n, m, p = 4, 4, 8
    rank = 98
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Inner Strassen coefficients (2x2)
    a_coeff = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, -1, 0]  # M7
    ]
    b_coeff = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    gamma = [
        [ [1, 0, 0, 1, -1, 0, 1], # c00
          [0, 0, 1, 0, 1, 0, 0]   # c01
        ],
        [ [0, 1, 0, 1, 0, 0, 0], # c10
          [1, -1, 1, 0, 0, 1, 0]  # c11
        ]
    ]
    
    # Outer Strassen coefficients (4x4 blocks)
    u_outer = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, -1, 0]  # M7
    ]
    v_outer = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    delta_matrix = [
        [ [1, 0, 0, 1, -1, 0, 1], # c00
          [0, 0, 1, 0, 1, 0, 0]   # c01
        ],
        [ [0, 1, 0, 1, 0, 0, 0], # c10
          [1, -1, 1, 0, 0, 1, 0]  # c11
        ]
    ]

    for half in range(2):
        base_c = half * 49
        for s in range(7):
            for t in range(7):
                c = base_c + s * 7 + t
                
                # U
                for I in range(2):
                    for J in range(2):
                        u_val = u_outer[s][I*2+J]
                        if u_val == 0: continue
                        for i in range(2):
                            for k in range(2):
                                a_val = a_coeff[t][i*2+k]
                                if a_val == 0: continue
                                a_idx = (I*2 + i) * 4 + (J*2 + k)
                                U[a_idx, c] = u_val * a_val
                                
                # V
                for K in range(2):
                    for L in range(2):
                        v_val = v_outer[s][K*2+L]
                        if v_val == 0: continue
                        for k in range(2):
                            for j in range(2):
                                b_val = b_coeff[t][k*2+j]
                                if b_val == 0: continue
                                b_row = K*2 + k
                                b_col = half * 4 + L*2 + j
                                b_idx = b_row * 8 + b_col
                                V[b_idx, c] = v_val * b_val

                # W
                for I in range(2):
                    for L in range(2):
                        d_val = delta_matrix[I][L][s]
                        if d_val == 0: continue
                        for i in range(2):
                            for j in range(2):
                                g_val = gamma[i][j][t]
                                if g_val == 0: continue
                                w_row = half * 4 + L * 2 + j
                                w_col = I * 2 + i
                                w_idx = w_row * 4 + w_col
                                W[w_idx, c] = d_val * g_val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
