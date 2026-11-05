import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 decomposition
    # U_strassen: 4x7
    # V_strassen: 4x7
    # W_strassen: 4x7
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M1 = (A00 + A11)(B00 + B11)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1 # C00, C11
    
    # M2 = (A10 + A11)B00
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[1, 1] = 1; W_s[3, 1] = -1 # C10, C11
    
    # M3 = A00(B01 - B11)
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1 # C01, C11
    
    # M4 = A11(B10 - B00)
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1 # C00, C10
    
    # M5 = (A00 + A01)B11
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[2, 4] = 1 # C00, C10
    
    # M6 = (A10 - A00)(B00 + B01)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[2, 5] = 1 # C11
    
    # M7 = (A01 - A11)(B10 + B11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1 # C00
    
    # Schoolbook 3x3 decomposition
    # U_t: 9x27
    # V_t: 9x27
    # W_t: 9x27
    U_t = np.zeros((9, 27))
    V_t = np.zeros((9, 27))
    W_t = np.zeros((9, 27))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_t[i * 3 + j, col] = 1
                V_t[j * 3 + k, col] = 1
                W_t[k * 3 + i, col] = 1
                col += 1
                
    # Combine
    rank_s = 7
    rank_t = 27
    total_rank = rank_s * rank_t
    
    U = np.zeros((36, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    for s in range(rank_s):
        for t in range(rank_t):
            r = s * rank_t + t
            
            # Fill U
            # U[3*a + b, 3*c + d, r] = U_s[a*2+c, s] * U_t[b*3+d, t]
            for a in range(2):
                for c in range(2):
                    val_s = U_s[a * 2 + c, s]
                    if val_s != 0:
                        for b in range(3):
                            for d in range(3):
                                val_t = U_t[b * 3 + d, t]
                                if val_t != 0:
                                    i = 3 * a + b
                                    j = 3 * c + d
                                    U[i * 6 + j, r] += val_s * val_t
                                    
            # Fill V
            # V[3*c + d, 3*e + f, r] = V_s[c*2+e, s] * V_t[d*3+f, t]
            for c in range(2):
                for e in range(2):
                    val_s = V_s[c * 2 + e, s]
                    if val_s != 0:
                        for d in range(3):
                            for f in range(3):
                                val_t = V_t[d * 3 + f, t]
                                if val_t != 0:
                                    j = 3 * c + d
                                    k = 3 * e + f
                                    V[j * 6 + k, r] += val_s * val_t
                                    
            # Fill W
            # W[3*e + f, 3*a + b, r] = W_s[e*2+a, s] * W_t[f*3+b, t]
            # Index in W is k*6 + i. k = 3e+f, i = 3a+b.
            for e in range(2):
                for a in range(2):
                    val_s = W_s[e * 2 + a, s]
                    if val_s != 0:
                        for f in range(3):
                            for b in range(3):
                                val_t = W_t[f * 3 + b, t]
                                if val_t != 0:
                                    k = 3 * e + f
                                    i = 3 * a + b
                                    W[k * 6 + i, r] += val_s * val_t
                                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
