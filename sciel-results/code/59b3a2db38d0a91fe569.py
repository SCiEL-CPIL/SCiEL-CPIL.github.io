import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # u: [A00, A01, A10, A11]
    # v: {(j, s): coeff} where s is local col (0 or 1)
    # w: {(i, s): coeff} where s is local col (0 or 1)
    
    terms = [
        # M1
        ([1, 0, 0, 1], {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        # M2
        ([0, 0, 1, 1], {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        # M3
        ([1, 0, 0, 0], {(0, 1): 1, (1, 1): -1}, {(0, 0): 1, (0, 1): 1}),
        # M4
        ([0, 0, 0, 1], {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        # M5
        ([1, 1, 0, 0], {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
        # M6
        ([0, 0, 1, -1], {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        # M7
        ([0, 1, 0, -1], {(1, 0): 1, (1, 1): 1}, {(0, 0): 1})
    ]
    
    for r in range(8):
        for t in range(7):
            R = r * 7 + t
            u_coeffs, v_coeffs, w_coeffs = terms[t]
            
            # Fill U
            for idx, val in enumerate(u_coeffs):
                U[idx][R] = val
                
            # Fill V
            for (j, s), val in v_coeffs.items():
                k = 2 * r + s
                v_idx = j * p + k
                V[v_idx][R] = val
                
            # Fill W
            for (i, s), val in w_coeffs.items():
                k = 2 * r + s
                w_idx = k * n + i
                W[w_idx][R] = val
                
    return {"U": U, "V": V, "W": W}
