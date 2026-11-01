import time

def construct():
    n, m, p = 2, 2, 6
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # U_coeffs[t][i][j]
    # V_coeffs[t][j][k_local]
    # W_coeffs[t][k_local][i]
    
    # Term 0: M1
    u0 = [[1, 0], [0, 1]]
    v0 = [[1, 0], [0, 1]]
    w0 = [[1, 1], [0, 0]] # C00 gets +1, C01 gets 0? No.
    # C00 = M1...
    # C01 = ...
    # C10 = ...
    # C11 = ...
    
    # Let's define W carefully.
    # W index is k*n + i.
    # k=0, i=0 -> index 0. C00.
    # k=1, i=0 -> index 1. C01.
    # k=0, i=1 -> index 2. C10.
    # k=1, i=1 -> index 3. C11.
    
    # M1: C00(+1), C11(-1)
    # M2: C10(+1), C11(-1)
    # M3: C01(+1)
    # M4: C00(-1), C01(+1)
    # M5: C10(+1), C11(-1)
    # M6: C00(-1), C11(+1)
    # M7: C00(+1), C01(+1), C10(-1)
    
    # Let's list them as lists of 4 values [C00, C01, C10, C11]
    w_coeffs = [
        [1, 0, 0, -1],  # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 0],   # M3
        [-1, 1, 0, 0],  # M4
        [0, 0, 1, -1],  # M5
        [-1, 0, 0, 1],  # M6
        [1, 1, -1, 0]   # M7
    ]
    
    u_coeffs = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [0, 1]], # M4
        [[1, 0], [1, 0]], # M5
        [[-1, 1], [0, 0]],# M6
        [[0, 0], [1, -1]] # M7
    ]
    
    v_coeffs = [
        [[1, 0], [0, 1]], # M1
        [[1, 0], [0, 0]], # M2
        [[0, 0], [1, -1]],# M3
        [[-1, 1], [0, 0]],# M4
        [[0, 0], [0, 1]], # M5
        [[1, 1], [0, 0]], # M6
        [[0, 1], [1, 0]]  # M7
    ]
    
    # Fill U, V, W
    for b in range(3):
        for t in range(7):
            c = b * 7 + t
            # U
            for i in range(2):
                for j in range(2):
                    U[i*m+j][c] = u_coeffs[t][i][j]
            
            # V
            for j in range(2):
                for k_local in range(2):
                    k_global = b * 2 + k_local
                    V[j*p+k_global][c] = v_coeffs[t][j][k_local]
            
            # W
            for k_local in range(2):
                for i in range(2):
                    k_global = b * 2 + k_local
                    # W index is k*n + i
                    W[k_global*n+i][c] = w_coeffs[t][k_local*2 + i]
                    
    return {"U": U, "V": V, "W": W}
