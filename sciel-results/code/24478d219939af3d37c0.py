import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # Block size 3, 2x2 blocks
    b = 3
    N = 2
    
    # Rank: 7 Strassen terms * 27 schoolbook terms per 3x3 mult
    rank = 7 * 27
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen definitions
    # r: 0..6
    # X_r: dict of (P, Q) -> coeff
    # Y_r: dict of (R, S) -> coeff
    # C_coeff: dict of (T, U) -> coeff for M_r in C_{T,U}
    
    strassen = [
        {
            'X': {(0,0): 1, (1,1): 1},
            'Y': {(0,0): 1, (1,1): 1},
            'C': {(0,0): 1, (1,1): 1}
        },
        {
            'X': {(1,0): 1, (1,1): 1},
            'Y': {(0,0): 1},
            'C': {(1,0): 1, (1,1): -1}
        },
        {
            'X': {(0,0): 1},
            'Y': {(0,1): 1, (1,1): -1},
            'C': {(0,1): 1, (1,1): 1}
        },
        {
            'X': {(1,1): 1},
            'Y': {(1,0): 1, (0,0): -1},
            'C': {(0,0): 1, (1,0): 1}
        },
        {
            'X': {(0,0): 1, (0,1): 1},
            'Y': {(1,1): 1},
            'C': {(0,0): -1, (0,1): 1}
        },
        {
            'X': {(1,0): 1, (0,0): -1},
            'Y': {(0,0): 1, (0,1): 1},
            'C': {(1,1): 1}
        },
        {
            'X': {(0,1): 1, (1,1): -1},
            'Y': {(1,0): 1, (1,1): 1},
            'C': {(0,0): 1}
        }
    ]
    
    # Iterate over Strassen terms
    for r in range(7):
        X_blocks = strassen[r]['X']
        Y_blocks = strassen[r]['Y']
        C_blocks = strassen[r]['C']
        
        # Iterate over schoolbook terms for 3x3 matrices
        # u, v, w in 0..2
        for u in range(3):
            for v in range(3):
                for w in range(3):
                    col = r * 27 + u * 9 + v * 3 + w
                    
                    # 1. Populate U
                    # X_{r, u, v} = sum_{(P,Q) in X_blocks} coeff * A_{P*3+u, Q*3+v}
                    for (P, Q), coeff_A in X_blocks.items():
                        i = P * 3 + u
                        j = Q * 3 + v
                        U[i * m + j][col] += coeff_A
                    
                    # 2. Populate V
                    # Y_{r, v, w} = sum_{(R,S) in Y_blocks} coeff * B_{R*3+v, S*3+w}
                    for (R, S), coeff_B in Y_blocks.items():
                        j = R * 3 + v
                        k = S * 3 + w
                        V[j * p + k][col] += coeff_B
                    
                    # 3. Populate W
                    # M_{r, u, w} contributes to C_{T, U} with coeff
                    # C_{T*3+u, U*3+w} += coeff * M_{r, u, w}
                    for (T, U_idx), coeff_C in C_blocks.items():
                        i = T * 3 + u
                        k = U_idx * 3 + w
                        W[k * n + i][col] += coeff_C
                        
    return {"U": U, "V": V, "W": W}
