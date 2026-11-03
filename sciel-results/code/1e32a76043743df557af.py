import time

def construct():
    n, m, p = 2, 3, 6
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Local decomposition for M_{2,3,2}
    # u: 6 entries (i*3+j)
    # v: 6 entries (j*2+k)
    # w: 4 entries (k*2+i)
    
    local_u = [
        [1, 0, 0, 1, 0, 0], # M1: A00+A11
        [0, 0, 0, 1, 1, 0], # M2: A10+A11
        [1, 0, 0, 0, 0, 0], # M3: A00
        [0, 0, 0, 1, 0, 0], # M4: A11
        [1, 1, 0, 0, 0, 0], # M5: A00+A01
        [-1, 0, 0, 0, 1, 0], # M6: A10-A00
        [0, 1, 0, -1, 0, 0], # M7: A01-A11
        [0, 0, 1, 0, 0, 0], # M8: A02
        [0, 0, 1, 0, 0, 0], # M9: A02
        [0, 0, 0, 0, 0, 1], # M10: A12
        [0, 0, 0, 0, 0, 1], # M11: A12
    ]
    
    local_v = [
        [1, 0, 0, 1, 0, 0], # M1: B00+B11
        [1, 0, 0, 0, 0, 0], # M2: B00
        [0, 1, 0, 0, -1, 0], # M3: B01-B11
        [-1, 0, 1, 0, 0, 0], # M4: B10-B00
        [0, 0, 0, 0, 1, 0], # M5: B11
        [1, 1, 0, 0, 0, 0], # M6: B00+B01
        [0, 0, 1, 0, 1, 0], # M7: B10+B11
        [0, 0, 1, 0, 0, 0], # M8: B20
        [0, 0, 0, 0, 1, 0], # M9: B21
        [0, 0, 1, 0, 0, 0], # M10: B20
        [0, 0, 0, 0, 1, 0], # M11: B21
    ]
    
    local_w = [
        [1, 0, 0, 1], # M1: C00+C11
        [0, 1, 0, 1], # M2: C10+C11
        [0, 0, 1, 1], # M3: C01+C11
        [1, 1, 0, 0], # M4: C00+C10
        [1, 0, -1, 0], # M5: C00-C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0], # M7: C00
        [1, 0, 0, 0], # M8: C00
        [0, 0, 1, 0], # M9: C01
        [0, 1, 0, 0], # M10: C10
        [0, 0, 0, 1], # M11: C11
    ]

    for b in range(3):
        col_offset = b * 11
        for t in range(11):
            col = col_offset + t
            
            # Fill U
            for i in range(2):
                for j in range(3):
                    U[i * 3 + j][col] = local_u[t][i * 3 + j]
            
            # Fill V
            for j in range(3):
                for k_local in range(2):
                    k_global = 2 * b + k_local
                    V[j * 6 + k_global][col] = local_v[t][j * 2 + k_local]
            
            # Fill W
            for i in range(2):
                for k_local in range(2):
                    k_global = 2 * b + k_local
                    w_index = k_global * 2 + i
                    W[w_index][col] = local_w[t][k_local * 2 + i]

    return {"U": U, "V": V, "W": W}
