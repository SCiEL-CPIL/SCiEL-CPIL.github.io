import numpy as np

def construct():
    n, m, p = 4, 4, 7
    
    # Strassen 2x2x2 decomposition
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, 0, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 1]
    ]
    
    # Build 4x4x4 decomposition via Kronecker product of two Strassen decomps
    # A index: (i1, i2, j1, j2) -> flat index i1*4 + i2*2 + j1*2 + j2? 
    # Let's use: A[i,j] where i = i1*2+i2, j = j1*2+j2
    # Flat A index: i*m + j = (i1*2+i2)*4 + (j1*2+j2)
    # Flat B index: j*p + k = (j1*2+j2)*4 + k for 4x4x4 part
    # Flat C index: k*n + i = k*4 + (i1*2+i2)
    
    # For Kronecker: rank = 7*7 = 49
    # U4[i1,i2,j1,j2, r1, r2] = U2[i1,j1, r1] * U2[i2,j2, r2]
    # V4[j1,j2,k1,k2, r1, r2] = V2[j1,k1, r1] * V2[j2,k2, r2]  (for 4x4x4, k is also 2x2)
    # W4[k1,k2,i1,i2, r1, r2] = W2[k1,i1, r1] * W2[k2,i2, r2]
    
    # But we need 4x4x7, not 4x4x4. Let's do 4x4x4 (rank 49) + 4x4x3 (rank 48) = 97
    
    rank = 97
    U = [[0] * rank for _ in range(n * m)]  # 16 x 97
    V = [[0] * rank for _ in range(m * p)]  # 28 x 97
    W = [[0] * rank for _ in range(p * n)]  # 28 x 97
    
    # Part 1: 4x4x4 using Strassen^2, rank 49
    # B indices for first 4 columns: j*p + k where k=0..3
    # C indices for first 4 columns: k*n + i where k=0..3
    
    col = 0
    for r1 in range(7):
        for r2 in range(7):
            # U part: for each (i1,i2,j1,j2)
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u_val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if u_val == 0:
                                continue
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U[i * m + j][col] += u_val
            # V part: for each (j1,j2,k) where k=0..3 for 4x4x4
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(2):
                            v_val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                            if v_val == 0:
                                continue
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            V[j * p + k][col] += v_val
            # W part: for each (k,i) where k=0..3
            for k1 in range(2):
                for k2 in range(2):
                    for i1 in range(2):
                        for i2 in range(2):
                            w_val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                            if w_val == 0:
                                continue
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            W[k * n + i][col] += w_val
            col += 1
    
    # Part 2: 4x4x3 schoolbook, rank 48
    # For k=4,5,6 (the extra 3 columns)
    for i in range(n):
        for j in range(m):
            for k in range(4, 7):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
