import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M0 = (x00+x11)(y00+y11)
    # M1 = (x10+x11)y00
    # M2 = x00(y01-y11)
    # M3 = x11(y10-y00)
    # M4 = (x00+x01)y11
    # M5 = (x10-x00)(y00+y01)
    # M6 = (x01-x11)(y10+y11)
    
    # Z00 = M0 - M3 + M4 + M6
    # Z01 = M2 + M4
    # Z10 = M1 + M3
    # Z11 = M0 - M1 + M2 + M5
    
    # Map of term index to (coeffs for A, coeffs for B, coeffs for C)
    # A coeffs: dict {(u,v): coeff}
    # B coeffs: dict {(v,w): coeff}
    # C coeffs: dict {(u,w): coeff}
    
    strassen_terms = [
        {
            'A': {(0,0): 1, (1,1): 1},
            'B': {(0,0): 1, (1,1): 1},
            'C': {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1} # M0 contributes to Z00 and Z11
        },
        {
            'A': {(1,0): 1, (1,1): 1},
            'B': {(0,0): 1},
            'C': {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1} # M1 contributes to Z10 (+1) and Z11 (-1)
        },
        {
            'A': {(0,0): 1},
            'B': {(0,1): 1, (1,1): -1},
            'C': {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1} # M2 contributes to Z01 (+1) and Z11 (+1)
        },
        {
            'A': {(1,1): 1},
            'B': {(1,0): 1, (0,0): -1},
            'C': {(0,0): -1, (0,1): 0, (1,0): 1, (1,1): 0} # M3 contributes to Z00 (-1) and Z10 (+1)
        },
        {
            'A': {(0,0): 1, (0,1): 1},
            'B': {(1,1): 1},
            'C': {(0,0): 1, (0,1): 1, (1,0): 0, (1,1): 0} # M4 contributes to Z00 (+1) and Z01 (+1)
        },
        {
            'A': {(1,0): 1, (0,0): -1},
            'B': {(0,0): 1, (0,1): 1},
            'C': {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1} # M5 contributes to Z11 (+1)
        },
        {
            'A': {(0,1): 1, (1,1): -1},
            'B': {(1,0): 1, (1,1): 1},
            'C': {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0} # M6 contributes to Z00 (+1)
        }
    ]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for term in strassen_terms:
                    # Fill U
                    for (u, v), coeff in term['A'].items():
                        row_u = (i * 2 + u) * 6 + (j * 2 + v)
                        U[row_u][col] += coeff
                        
                    # Fill V
                    for (v, w), coeff in term['B'].items():
                        row_v = (j * 2 + v) * 6 + (k * 2 + w)
                        V[row_v][col] += coeff
                        
                    # Fill W
                    for (u, w), coeff in term['C'].items():
                        row_w = (k * 2 + w) * 6 + (i * 2 + u)
                        W[row_w][col] += coeff
                        
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
