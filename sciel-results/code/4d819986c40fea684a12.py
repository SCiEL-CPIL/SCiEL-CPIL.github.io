import numpy as np

def construct():
    n, m, p = 2, 3, 8
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    strassen_U = {
        0: {0: 1, 3: 1},  # M1: A00, A11
        1: {2: 1, 3: 1},  # M2: A10, A11
        2: {0: 1},        # M3: A00
        3: {3: 1},        # M4: A11
        4: {0: 1, 1: 1},  # M5: A00, A01
        5: {2: 1, 0: -1}, # M6: A10, -A00
        6: {1: 1, 3: -1}  # M7: A01, -A11
    }
    strassen_V = {
        0: {0: 1, 3: 1},  # M1: B00, B11
        1: {0: 1},        # M2: B00
        2: {1: 1, 3: -1}, # M3: B01, -B11
        3: {2: 1, 0: -1}, # M4: B10, -B00
        4: {3: 1},        # M5: B11
        5: {0: 1, 1: 1},  # M6: B00, B01
        6: {2: 1, 3: 1}   # M7: B10, B11
    }
    strassen_W = {
        0: {0: 1, 3: 1},  # M1: C00, C11
        1: {2: 1, 3: -1}, # M2: C10, -C11
        2: {1: 1, 3: 1},  # M3: C01, C11
        3: {0: 1, 2: 1},  # M4: C00, C10
        4: {0: -1, 1: 1}, # M5: -C00, C01
        5: {3: 1},        # M6: C11
        6: {0: 1}         # M7: C00
    }
    
    col = 0
    for block in range(2):
        k_offset = block * 4
        
        # Part 1: 2x2x4 using j=0,1
        # This consists of two 2x2x2 Strassen blocks
        for sub_block in range(2):
            k_sub_offset = sub_block * 2
            for c in range(7):
                # U
                for u_idx, val in strassen_U[c].items():
                    i = u_idx // 2
                    j = u_idx % 2
                    U[i * m + j][col] += val
                # V
                for v_idx, val in strassen_V[c].items():
                    j = v_idx // 2
                    k_local = v_idx % 2
                    k_global = k_offset + k_sub_offset + k_local
                    V[j * p + k_global][col] += val
                # W
                for w_idx, val in strassen_W[c].items():
                    k_local = w_idx // 2
                    i = w_idx % 2
                    k_global = k_offset + k_sub_offset + k_local
                    W[k_global * n + i][col] += val
                col += 1
        
        # Part 2: 2x1x4 using j=2
        # Rank 1
        for i in range(n):
            U[i * m + 2][col] = 1
        for k_local in range(4):
            k_global = k_offset + k_local
            V[2 * p + k_global][col] = 1
        for k_global in range(k_offset, k_offset + 4):
            for i in range(n):
                W[k_global * n + i][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
