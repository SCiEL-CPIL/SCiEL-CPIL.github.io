import time

DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for one 2x2 block
    # l=0..6 corresponds to M1..M7
    # u[i, j], v[j, k_local], w[k_local, i_prime]
    # k_local is 0 or 1 (relative to 2q)
    
    strassen_coeffs = [
        # M1: (A00+A11)(B00+B11) -> C00 +=, C11 +=
        {'u': {(0,0): 1, (1,1): 1}, 'v': {(0,0): 1, (1,1): 1}, 'w': {(0,0): 1, (1,1): 1}},
        # M2: (A10+A11)B00 -> C01 -=, C11 +=
        {'u': {(1,0): 1, (1,1): 1}, 'v': {(0,0): 1}, 'w': {(0,1): -1, (1,1): 1}},
        # M3: A00(B01-B11) -> C10 +=, C11 +=
        {'u': {(0,0): 1}, 'v': {(0,1): 1, (1,1): -1}, 'w': {(1,0): 1, (1,1): 1}},
        # M4: A11(B10-B00) -> C00 +=, C01 +=
        {'u': {(1,1): 1}, 'v': {(1,0): 1, (0,0): -1}, 'w': {(0,0): 1, (0,1): 1}},
        # M5: (A00+A01)B11 -> C00 -=, C10 +=
        {'u': {(0,0): 1, (0,1): 1}, 'v': {(1,1): 1}, 'w': {(0,0): -1, (1,0): 1}},
        # M6: (A10-A00)(B00+B01) -> C11 +=
        {'u': {(1,0): 1, (0,0): -1}, 'v': {(0,0): 1, (0,1): 1}, 'w': {(1,1): 1}},
        # M7: (A01-A11)(B10+B11) -> C00 +=
        {'u': {(0,1): 1, (1,1): -1}, 'v': {(1,0): 1, (1,1): 1}, 'w': {(0,0): 1}},
    ]
    
    for q in range(8):
        k0 = 2 * q
        k1 = 2 * q + 1
        for l in range(7):
            r = 7 * q + l
            coeff = strassen_coeffs[l]
            
            for (i, j), val in coeff['u'].items():
                U[i * m + j][r] = val
                
            for (j, kl), val in coeff['v'].items():
                k = k0 + kl
                V[j * p + k][r] = val
                
            for (kl, ip), val in coeff['w'].items():
                k = k0 + kl
                W[k * n + ip][r] = val
                
    return {"U": U, "V": V, "W": W}
