import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # U_local: 4 entries (a, b, c, d)
    # V_local: 4 entries (e, f, g, h)
    # W_local: 4 entries (C00, C01, C10, C11)
    
    terms = [
        # P1
        {'U': [1, 0, 0, 1], 'V': [1, 0, 0, 1], 'W': [1, 0, 0, 1]},
        # P2
        {'U': [0, 0, 1, 1], 'V': [1, 0, 0, 0], 'W': [0, 0, 1, -1]},
        # P3
        {'U': [1, 0, 0, 0], 'V': [0, 1, 0, -1], 'W': [0, 1, 0, 1]},
        # P4
        {'U': [0, 0, 0, 1], 'V': [-1, 0, 1, 0], 'W': [1, 0, 1, 0]},
        # P5
        {'U': [1, 1, 0, 0], 'V': [0, 0, 0, 1], 'W': [-1, 1, 0, 0]},
        # P6
        {'U': [-1, 0, 1, 0], 'V': [1, 1, 0, 0], 'W': [0, 0, 0, 1]},
        # P7
        {'U': [0, 1, 0, -1], 'V': [0, 0, 1, 1], 'W': [1, 0, 0, 0]},
    ]
    
    for b in range(3):
        for t_idx in range(7):
            col = b * 7 + t_idx
            term = terms[t_idx]
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    row_u = i * m + j
                    U[row_u][col] = term['U'][i * 2 + j]
            
            # Fill V
            for j in range(2):
                for k_local in range(2):
                    k_global = b * 2 + k_local
                    row_v = j * p + k_global
                    V[row_v][col] = term['V'][j * 2 + k_local]
            
            # Fill W
            for k_local in range(2):
                for i in range(2):
                    k_global = b * 2 + k_local
                    row_w = k_global * n + i
                    W[row_w][col] = term['W'][k_local * 2 + i]
                    
    return {"U": U, "V": V, "W": W}
