import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 factors
    # U_str: 4x7
    # V_str: 4x7
    # W_str: 4x7
    
    U_str = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    
    V_str = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    
    W_str = [
        [1, 0, 0, -1, 0, -1, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, 0, -1, 0, -1, 0, 1]  # C11
    ]
    
    # Construct 2x2x4 decomposition
    # U_224: 4x14 = [U_str, U_str]
    # V_224: 8x14
    # W_224: 8x14 = [W_str, W_str]
    
    # We will fill U, V, W for the two blocks (A0*Btop and A1*Bbot)
    
    # Block 1: A0 * Btop
    # A0 corresponds to i in 0,1; j in 0,1
    # Btop corresponds to j in 0,1; k in 0,1,2,3
    # C corresponds to k in 0,1,2,3; i in 0,1
    
    for r in range(7):
        # U part
        for i in range(2):
            for j in range(2):
                U[i * m + j][r] = U_str[i * 2 + j][r]
                
        # V part
        for j in range(2):
            for k in range(4):
                # Local V index j*4+k
                # V_str index j*2 + (k%2) ? No.
                # V_str is 2x2. V_224 is 2x4.
                # V_224[j, k] for k=0,1 comes from V_str[j, k]
                # V_224[j, k] for k=2,3 comes from V_str[j, k-2]
                if k < 2:
                    V[j * p + k][r] = V_str[j * 2 + k][r]
                else:
                    V[j * p + k][r] = V_str[j * 2 + (k - 2)][r]
                    
        # W part
        for k in range(4):
            for i in range(2):
                # W_224 index k*2+i
                # W_str index k%2 * 2 + i ?
                # W_224[k, i] for k=0,1 comes from W_str[k, i]
                # W_224[k, i] for k=2,3 comes from W_str[k-2, i]
                if k < 2:
                    W[k * n + i][r] = W_str[k * 2 + i][r]
                else:
                    W[k * n + i][r] = W_str[(k - 2) * 2 + i][r]
                    
    # Block 2: A1 * Bbot
    # A1 corresponds to i in 0,1; j in 2,3
    # Bbot corresponds to j in 2,3; k in 0,1,2,3
    # C corresponds to k in 0,1,2,3; i in 0,1
    
    for r in range(7, 14):
        r_local = r - 7
        
        # U part
        for i in range(2):
            for j in range(2):
                # Full j = j_local + 2
                U[i * m + (j + 2)][r] = U_str[i * 2 + j][r_local]
                
        # V part
        for j_local in range(2):
            for k in range(4):
                # Full j = j_local + 2
                if k < 2:
                    V[(j_local + 2) * p + k][r] = V_str[j_local * 2 + k][r_local]
                else:
                    V[(j_local + 2) * p + k][r] = V_str[j_local * 2 + (k - 2)][r_local]
                    
        # W part
        for k in range(4):
            for i in range(2):
                if k < 2:
                    W[k * n + i][r] += W_str[k * 2 + i][r_local]
                else:
                    W[k * n + i][r] += W_str[(k - 2) * 2 + i][r_local]

    return {"U": U, "V": V, "W": W}
