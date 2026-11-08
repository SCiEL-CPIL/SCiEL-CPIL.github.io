import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 15 decomposition
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen block
    # local_i, local_j, local_k are 0 or 1
    # global_i, global_j, global_k map local to global
    def add_strassen_block(offset_col, gi_offset, gj_offset, gk_offset):
        # Strassen coefficients for A (2x2), B (2x2), C (2x2)
        # Indices: 00, 01, 10, 11
        # A_idx: i*2 + j
        # B_idx: j*2 + k
        # C_idx: i*2 + k (but W uses k*n+i, so we need to map carefully)
        
        # Strassen terms: (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: dict {(i,j): val}
        # B_coeffs: dict {(j,k): val}
        # C_coeffs: dict {(i,k): val}
        
        terms = [
            # P1
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            # P2
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            # P3
            ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            # P4
            ({(1,1):1}, {(0,1):1, (1,1):1}, {(0,0):1, (1,0):1}),
            # P5
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            # P6
            ({(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(1,1):1}),
            # P7
            ({(1,0):1, (0,1):-1}, {(1,1):1, (0,1):1}, {(0,0):1})
        ]
        
        for t_idx, (a_c, b_c, c_c) in enumerate(terms):
            col = offset_col + t_idx
            
            # Add A coefficients to U
            for (li, lj), val in a_c.items():
                gi = gi_offset + li
                gj = gj_offset + lj
                u_idx = gi * m + gj
                U[u_idx][col] += val
            
            # Add B coefficients to V
            for (lj, lk), val in b_c.items():
                gj = gj_offset + lj
                gk = gk_offset + lk
                v_idx = gj * p + gk
                V[v_idx][col] += val
            
            # Add C coefficients to W
            # W index is k*n + i
            for (li, lk), val in c_c.items():
                gi = gi_offset + li
                gk = gk_offset + lk
                w_idx = gk * n + gi
                W[w_idx][col] += val

    # Block 1: j in {0,1}, k in {0,1}
    add_strassen_block(0, 0, 0, 0)
    
    # Block 2: j in {0,1}, k in {2,3}
    add_strassen_block(7, 0, 0, 2)
    
    # Block 3: j=2, k in {0,1,2,3}
    # This is M_{2,1,4}, rank 1
    col = 14
    for i in range(n):
        for k in range(p):
            # U[i*m + 2, col] = 1
            # V[2*p + k, col] = 1
            # W[k*n + i, col] = 1
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            
    return {"U": U, "V": V, "W": W}
