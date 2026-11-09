import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 6x6 matrix multiplication using 3x3 grid of 2x2 Strassen blocks
    # Total rank: 27 block multiplications * 7 Strassen terms = 189
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2 decomposition terms
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    terms = [
        # M1: (a+d)(e+h)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2: (c+d)e
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3: a(f-h)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4: d(g-e)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5: (a+b)h
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6: (c-a)(e+f)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7: (b-d)(g+h)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Iterate over 3x3 block grid
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for u_coeffs, v_coeffs, w_coeffs in terms:
                    for local_idx, coeff in u_coeffs.items():
                        r, c = divmod(local_idx, 2)
                        global_r = 2 * i + r
                        global_c = 2 * k + c
                        U[global_r * m + global_c][col] += coeff
                    
                    for local_idx, coeff in v_coeffs.items():
                        r, c = divmod(local_idx, 2)
                        global_r = 2 * k + r
                        global_c = 2 * j + c
                        V[global_r * p + global_c][col] += coeff
                    
                    for local_idx, coeff in w_coeffs.items():
                        r, c = divmod(local_idx, 2)
                        global_r = 2 * i + r
                        global_c = 2 * j + c
                        W[global_c * n + global_r][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
