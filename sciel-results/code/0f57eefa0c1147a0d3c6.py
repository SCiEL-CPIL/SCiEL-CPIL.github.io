import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    rank = 7 * 27
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    steps = [
        {
            'A': {(0,0): 1, (1,1): 1},
            'B': {(0,0): 1, (1,1): 1},
            'C': {(0,0): 1, (1,1): 1}
        },
        {
            'A': {(1,0): 1, (1,1): 1},
            'B': {(0,0): 1},
            'C': {(1,0): 1, (1,1): -1}
        },
        {
            'A': {(0,0): 1},
            'B': {(0,1): 1, (1,1): -1},
            'C': {(0,1): 1, (1,1): 1}
        },
        {
            'A': {(1,1): 1},
            'B': {(1,0): 1, (0,0): -1},
            'C': {(0,0): 1, (1,0): 1}
        },
        {
            'A': {(0,0): 1, (0,1): 1},
            'B': {(1,1): 1},
            'C': {(0,0): -1, (0,1): 1}
        },
        {
            'A': {(1,0): 1, (0,0): -1},
            'B': {(0,0): 1, (0,1): 1},
            'C': {(1,1): 1}
        },
        {
            'A': {(0,1): 1, (1,1): -1},
            'B': {(1,0): 1, (1,1): 1},
            'C': {(0,0): 1}
        }
    ]
    
    col = 0
    for s in steps:
        A_coeffs = s['A']
        B_coeffs = s['B']
        C_coeffs = s['C']
        
        for u in range(3):
            for v in range(3):
                for w in range(3):
                    # U part
                    for (a, b), coeff_a in A_coeffs.items():
                        i = a * 3 + u
                        j = b * 3 + w
                        U[i * 6 + j, col] += coeff_a
                    
                    # V part
                    for (b, g), coeff_b in B_coeffs.items():
                        j = b * 3 + w
                        k = g * 3 + v
                        V[j * 6 + k, col] += coeff_b
                    
                    # W part
                    for (a, g), coeff_c in C_coeffs.items():
                        k = g * 3 + v
                        i = a * 3 + u
                        W[k * 6 + i, col] += coeff_c
                        
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
