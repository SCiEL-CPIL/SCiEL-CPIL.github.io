import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 88
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 7 factors: (a_coeffs, b_coeffs, c_coeffs)
    s7 = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    for cblk in range(2):
        for f in s7:
            ac, bc, cc = f
            for sp in range(4):
                if sp == 0:
                    ao_i, ao_j = 0, 0
                    bo_i, bo_j = 0, 0
                    co_i, co_j = 0, 0
                elif sp == 1:
                    ao_i, ao_j = 0, 0
                    bo_i, bo_j = 0, 2
                    co_i, co_j = 0, 2
                elif sp == 2:
                    ao_i, ao_j = 0, 2
                    bo_i, bo_j = 2, 0
                    co_i, co_j = 0, 0
                else:
                    ao_i, ao_j = 0, 2
                    bo_i, bo_j = 2, 2
                    co_i, co_j = 0, 2
                
                for (ai, aj), au in ac.items():
                    ui = (ai + ao_i) * 4 + (aj + ao_j)
                    U[ui][col] += au
                
                for (bi, bj), bv in bc.items():
                    vi = (bi + bo_i) * 8 + (bj + bo_j + 4 * cblk)
                    V[vi][col] += bv
                
                for (ci, cj), cw in cc.items():
                    wi = (cj + co_j + 4 * cblk) * 3 + (ci + co_i)
                    W[wi][col] += cw
                
                col += 1
    
    for cblk in range(2):
        for k in range(4):
            for j in range(4):
                ui = 2 * 4 + k
                vi = k * 8 + (j + 4 * cblk)
                wi = (j + 4 * cblk) * 3 + 2
                U[ui][col] = 1
                V[vi][col] = 1
                W[wi][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
