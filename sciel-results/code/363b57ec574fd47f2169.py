import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen definitions
    # M1: (x00+x11)(y00+y11)
    # M2: (x10+x11)y00
    # M3: x00(y10-y11)
    # M4: x11(y01-y00)
    # M5: (x00+x01)y11
    # M6: (x10-x00)(y00+y01)
    # M7: (x01-x11)(y10+y11)
    
    # Coefficients for U (x), V (y), W (z)
    # U indices: (i, j_loc)
    # V indices: (j_loc, k_loc)
    # W indices: (k_loc, i)
    
    strassen_terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u_coeffs: list of (i, j_loc, coeff)
        # v_coeffs: list of (j_loc, k_loc, coeff)
        # w_coeffs: list of (k_loc, i, coeff)
        
        # M1
        ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ], [ (0,0,1), (0,1,1), (1,0,-1), (1,1,-1) ]), # Z00=1, Z01=1, Z10=-1, Z11=-1? No.
        # Z00 = M1 - M4 + M5 - M7
        # Z01 = M3 + M7
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        # Let's just use the formulas for Z directly.
        # It's easier to define the contribution of each M to Z.
    ]
    
    # It is better to define the 7 terms and their contributions to Z.
    # Term 1 (M1):
    # U: (0,0,1), (1,1,1)
    # V: (0,0,1), (1,1,1)
    # W: (0,0,1), (1,1,1), (0,1,0), (1,0,0) -> Wait, Z00 gets +1, Z11 gets +1.
    # Z00 = M1... so W(0,0) += 1
    # Z11 = M1... so W(1,1) += 1
    
    # Term 2 (M2):
    # U: (1,0,1), (1,1,1)
    # V: (0,0,1)
    # W: (1,0,1) # Z10 += 1, Z11 -= 1
    
    # Term 3 (M3):
    # U: (0,0,1)
    # V: (1,0,1), (1,1,-1)
    # W: (0,1,1), (1,1,1) # Z01 += 1, Z11 += 1
    
    # Term 4 (M4):
    # U: (1,1,1)
    # V: (0,1,1), (0,0,-1)
    # W: (0,0,-1), (1,0,1) # Z00 -= 1, Z10 += 1
    
    # Term 5 (M5):
    # U: (0,0,1), (0,1,1)
    # V: (1,1,1)
    # W: (0,0,1) # Z00 += 1
    
    # Term 6 (M6):
    # U: (1,0,1), (0,0,-1)
    # V: (0,0,1), (0,1,1)
    # W: (1,1,1) # Z11 += 1
    
    # Term 7 (M7):
    # U: (0,1,1), (1,1,-1)
    # V: (1,0,1), (1,1,1)
    # W: (0,0,-1), (0,1,1) # Z00 -= 1, Z01 += 1

    terms = [
        # (u, v, w)
        ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]),
        ([ (1,0,1), (1,1,1) ], [ (0,0,1) ], [ (1,0,1), (1,1,-1) ]),
        ([ (0,0,1) ], [ (1,0,1), (1,1,-1) ], [ (0,1,1), (1,1,1) ]),
        ([ (1,1,1) ], [ (0,1,1), (0,0,-1) ], [ (0,0,-1), (1,0,1) ]),
        ([ (0,0,1), (0,1,1) ], [ (1,1,1) ], [ (0,0,1) ]),
        ([ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ], [ (1,1,1) ]),
        ([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ], [ (0,0,-1), (0,1,1) ])
    ]

    for part in range(2):
        for sub in range(4):
            for t in range(7):
                u_defs, v_defs, w_defs = terms[t]
                
                # Fill U
                for i_loc, j_loc, coeff in u_defs:
                    u_idx = i_loc * m + (part * 2 + j_loc)
                    U[u_idx][col] += coeff
                    
                # Fill V
                for j_loc, k_loc, coeff in v_defs:
                    v_idx = (part * 2 + j_loc) * p + (sub * 2 + k_loc)
                    V[v_idx][col] += coeff
                    
                # Fill W
                for k_loc, i_loc, coeff in w_defs:
                    w_idx = (sub * 2 + k_loc) * n + i_loc
                    W[w_idx][col] += coeff
                    
                col += 1

    return {"U": U, "V": V, "W": W}
