import time

DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 8 * 7 = 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # s is 0..6 corresponding to M1..M7
    # A_coeffs[s][i][j]
    # B_coeffs[s][j][k]
    # C_coeffs[s][i][k]
    
    A_coeffs = [
        [[1, 0], [0, 1]], # M1: A00, A11
        [[0, 0], [1, 1]], # M2: A10, A11
        [[1, 0], [0, 0]], # M3: A00
        [[0, 0], [0, 1]], # M4: A11
        [[1, 1], [0, 0]], # M5: A00, A01
        [[-1, 0], [1, 0]],# M6: -A00, A10
        [[0, 1], [0, -1]] # M7: A01, -A11
    ]
    
    B_coeffs = [
        [[1, 0], [0, 1]], # M1: B00, B11
        [[1, 0], [0, 0]], # M2: B00
        [[0, 0], [1, -1]],# M3: B10, -B11
        [[-1, 1], [0, 0]],# M4: -B00, B01
        [[0, 0], [0, 1]], # M5: B11
        [[1, 1], [0, 0]], # M6: B00, B01
        [[0, 0], [1, 1]]  # M7: B10, B11
    ]
    
    C_coeffs = [
        [[1, 0], [0, 1]], # M1: C00, C11
        [[0, 0], [1, -1]],# M2: C10, -C11
        [[0, 1], [0, 1]], # M3: C01, C11
        [[-1, 0], [1, 0]],# M4: -C00, C10
        [[1, 0], [0, 0]], # M5: C00
        [[0, 0], [0, 1]], # M6: C11
        [[-1, 1], [0, 0]] # M7: -C00, C01
    ]
    
    num_blocks = p // 2
    for l in range(num_blocks):
        for s in range(7):
            r = l * 7 + s
            # Fill U
            for i in range(n):
                for j in range(m):
                    U[i * m + j][r] = A_coeffs[s][i][j]
            
            # Fill V
            for j in range(m):
                for k_local in range(2):
                    k_global = l * 2 + k_local
                    V[j * p + k_global][r] = B_coeffs[s][j][k_local]
            
            # Fill W
            for i in range(n):
                for k_local in range(2):
                    k_global = l * 2 + k_local
                    W[k_global * n + i][r] = C_coeffs[s][i][k_local]
                    
    return {"U": U, "V": V, "W": W}
