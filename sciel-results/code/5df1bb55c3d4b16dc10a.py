import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U (A matrix part)
    # s=0: M1, s=1: M2, s=2: M3, s=3: M4, s=4: M5, s=5: M6, s=6: M7
    # Rows of U: (0,0), (0,1), (1,0), (1,1)
    U_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # V coefficients for each s and block r
    # V rows: j*p + k
    # j=0, k=2r (e)
    # j=0, k=2r+1 (f)
    # j=1, k=2r (g)
    # j=1, k=2r+1 (h)
    
    # W coefficients for each s and block r
    # W rows: k*n + i
    # k=2r, i=0 (C_0, 2r)
    # k=2r, i=1 (C_1, 2r)
    # k=2r+1, i=0 (C_0, 2r+1)
    # k=2r+1, i=1 (C_1, 2r+1)
    
    for r in range(8):
        for s in range(7):
            c = r * 7 + s
            
            # Fill U
            for idx, val in enumerate(U_coeffs[s]):
                if val != 0:
                    U[idx][c] = val
            
            # Fill V
            # Indices:
            # e: 0*16 + 2r
            # f: 0*16 + 2r+1
            # g: 1*16 + 2r
            # h: 1*16 + 2r+1
            
            if s == 0: # M1: e+h
                V[0 * p + 2*r][c] = 1
                V[1 * p + 2*r + 1][c] = 1
            elif s == 1: # M2: e
                V[0 * p + 2*r][c] = 1
            elif s == 2: # M3: f-h
                V[0 * p + 2*r + 1][c] = 1
                V[1 * p + 2*r + 1][c] = -1
            elif s == 3: # M4: g-e
                V[1 * p + 2*r][c] = 1
                V[0 * p + 2*r][c] = -1
            elif s == 4: # M5: h
                V[1 * p + 2*r + 1][c] = 1
            elif s == 5: # M6: e+f
                V[0 * p + 2*r][c] = 1
                V[0 * p + 2*r + 1][c] = 1
            elif s == 6: # M7: g+h
                V[1 * p + 2*r][c] = 1
                V[1 * p + 2*r + 1][c] = 1
            
            # Fill W
            # Indices:
            # C_0, 2r: 2r*2 + 0 = 4r
            # C_1, 2r: 2r*2 + 1 = 4r+1
            # C_0, 2r+1: (2r+1)*2 + 0 = 4r+2
            # C_1, 2r+1: (2r+1)*2 + 1 = 4r+3
            
            if s == 0: # M1: C_0, 2r + C_1, 2r+1
                W[4*r][c] = 1
                W[4*r + 3][c] = 1
            elif s == 1: # M2: C_1, 2r - C_1, 2r+1
                W[4*r + 1][c] = 1
                W[4*r + 3][c] = -1
            elif s == 2: # M3: C_0, 2r+1 + C_1, 2r+1
                W[4*r + 2][c] = 1
                W[4*r + 3][c] = 1
            elif s == 3: # M4: C_0, 2r + C_1, 2r
                W[4*r][c] = 1
                W[4*r + 1][c] = 1
            elif s == 4: # M5: -C_0, 2r + C_0, 2r+1
                W[4*r][c] = -1
                W[4*r + 2][c] = 1
            elif s == 5: # M6: C_1, 2r+1
                W[4*r + 3][c] = 1
            elif s == 6: # M7: C_0, 2r
                W[4*r][c] = 1
                
    return {"U": U, "V": V, "W": W}
