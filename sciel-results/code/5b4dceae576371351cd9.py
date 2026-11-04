import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 65
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U, V, W are 4x7 matrices (flattened 2x2)
    # Indices: (i,j) -> i*2+j
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # Term 1: (A11 + A22)(B11 + B22)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1 # C11, C22
    
    # Term 2: (A21 + A22)B11
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = -1 # C21, C22
    
    # Term 3: A11(B12 - B22)
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1 # C12, C22
    
    # Term 4: A22(B21 - B11)
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1 # C11, C21
    
    # Term 5: (A11 + A12)B22
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1 # C11, C12
    
    # Term 6: (A21 - A11)(B11 + B12)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1 # C22
    
    # Term 7: (A12 - A22)(B21 + B22)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1 # C11

    # Strassen block coefficients for 4x4x4
    # k=0..6
    # c_ij: coefficients for A blocks
    # d_kl: coefficients for B blocks
    # w_mn: coefficients for C blocks
    c = np.zeros((7, 4))
    d = np.zeros((7, 4))
    w = np.zeros((7, 4))
    
    # k=0: M1
    c[0, 0] = 1; c[0, 3] = 1
    d[0, 0] = 1; d[0, 3] = 1
    w[0, 0] = 1; w[0, 3] = 1
    
    # k=1: M2
    c[1, 2] = 1; c[1, 3] = 1
    d[1, 0] = 1
    w[1, 2] = 1; w[1, 3] = -1
    
    # k=2: M3
    c[2, 0] = 1
    d[2, 1] = 1; d[2, 3] = -1
    w[2, 1] = 1; w[2, 3] = 1
    
    # k=3: M4
    c[3, 3] = 1
    d[3, 2] = 1; d[3, 0] = -1
    w[3, 0] = 1; w[3, 2] = 1
    
    # k=4: M5
    c[4, 0] = 1; c[4, 1] = 1
    d[4, 3] = 1
    w[4, 0] = -1; w[4, 1] = 1
    
    # k=5: M6
    c[5, 2] = 1; c[5, 0] = -1
    d[5, 0] = 1; d[5, 1] = 1
    w[5, 3] = 1
    
    # k=6: M7
    c[6, 1] = 1; c[6, 3] = -1
    d[6, 2] = 1; d[6, 3] = 1
    w[6, 0] = 1

    col = 0
    for k in range(7):
        for s in range(7):
            for i in range(2):
                for j in range(2):
                    for p in range(2):
                        for q in range(2):
                            idx_U = (i * 2 + p) * 4 + (j * 2 + q)
                            if c[k, i * 2 + j] != 0 and U_s[p * 2 + q, s] != 0:
                                U[idx_U][col] += c[k, i * 2 + j] * U_s[p * 2 + q, s]
            for k_idx in range(2):
                for l in range(2):
                    for r in range(2):
                        for s_idx in range(2):
                            idx_V = (k_idx * 2 + r) * 5 + (l * 2 + s_idx)
                            if d[k, k_idx * 2 + l] != 0 and V_s[r * 2 + s_idx, s] != 0:
                                V[idx_V][col] += d[k, k_idx * 2 + l] * V_s[r * 2 + s_idx, s]
            for m in range(2):
                for n in range(2):
                    for p_out in range(2):
                        for r_out in range(2):
                            idx_W = (n * 2 + r_out) * 4 + (m * 2 + p_out)
                            if w[k, m * 2 + n] != 0 and W_s[p_out * 2 + r_out, s] != 0:
                                W[idx_W][col] += w[k, m * 2 + n] * W_s[p_out * 2 + r_out, s]
            col += 1
            
    # 4x4x1 part: A (4x4) * b (4x1)
    # Rank 16
    for i in range(4):
        for j in range(4):
            idx_U = i * 4 + j
            idx_V = j * 5 + 4 # 5th column of B
            idx_W = 4 * 4 + i # 5th row of C (index 4 in 0..4, but W is 5x4? No, W is p*n = 5*4=20)
            # W index: row k (0..4), col i (0..3)
            # C[k, i] = sum_j A[i, j] B[j, k]
            # Here k=4 (last col of B), so C[4, i] = sum_j A[i, j] B[j, 4]
            # W index for C[k, i] is k * n + i = 4 * 4 + i
            U[idx_U][col] = 1
            V[idx_V][col] = 1
            W[idx_W][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
