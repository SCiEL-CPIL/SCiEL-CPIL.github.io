import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 2x2 Strassen factors
    # u2[i*2+j, s], v2[j*2+k, s], w2[k*2+i, s]
    u2 = np.zeros((4, 7))
    v2 = np.zeros((4, 7))
    w2 = np.zeros((4, 7))
    
    # s=0: M1
    u2[0, 0] = 1; u2[3, 0] = 1
    v2[0, 0] = 1; v2[3, 0] = 1
    # w2 for C00, C01, C10, C11
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # w2 index is k*2 + i.
    # C00 (i=0, k=0) -> idx 0
    # C01 (i=0, k=1) -> idx 1
    # C10 (i=1, k=0) -> idx 2
    # C11 (i=1, k=1) -> idx 3
    
    # M1 (s=0)
    w2[0, 0] = 1 # C00
    w2[3, 0] = 1 # C11
    
    # M2 (s=1)
    u2[2, 1] = 1; u2[3, 1] = 1
    v2[0, 1] = 1
    w2[2, 1] = 1 # C10
    w2[3, 1] = -1 # C11
    
    # M3 (s=2)
    u2[0, 2] = 1
    v2[1, 2] = 1; v2[3, 2] = -1
    w2[1, 2] = 1 # C01
    w2[3, 2] = 1 # C11
    
    # M4 (s=3)
    u2[3, 3] = 1
    v2[2, 3] = 1; v2[0, 3] = -1
    w2[0, 3] = 1 # C00
    w2[2, 3] = 1 # C10
    
    # M5 (s=4)
    u2[0, 4] = 1; u2[1, 4] = 1
    v2[3, 4] = 1
    w2[0, 4] = -1 # C00
    w2[1, 4] = 1 # C01
    
    # M6 (s=5)
    u2[2, 5] = 1; u2[0, 5] = -1
    v2[0, 5] = 1; v2[1, 5] = 1
    w2[3, 5] = 1 # C11
    
    # M7 (s=6)
    u2[1, 6] = 1; u2[3, 6] = -1
    v2[2, 6] = 1; v2[3, 6] = 1
    w2[0, 6] = 1 # C00

    # Construct 4x4 factors
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    for s1 in range(7):
        for s2 in range(7):
            c = s1 * 7 + s2
            for i in range(4):
                for j in range(4):
                    i1, i2 = divmod(i, 2)
                    j1, j2 = divmod(j, 2)
                    U[i * 4 + j, c] = u2[i1 * 2 + j1, s1] * u2[i2 * 2 + j2, s2]
                    
                    for k in range(4):
                        j1, j2 = divmod(j, 2)
                        k1, k2 = divmod(k, 2)
                        V[j * 4 + k, c] = v2[j1 * 2 + k1, s1] * v2[j2 * 2 + k2, s2]
                        
                        # W index is k * n + i
                        k1, k2 = divmod(k, 2)
                        i1, i2 = divmod(i, 2)
                        W[k * 4 + i, c] = w2[k1 * 2 + i1, s1] * w2[k2 * 2 + i2, s2]
                        
    # Verification
    # Check if the decomposition is exact
    # We can test with a few random matrices
    import random
    random.seed(42)
    for _ in range(5):
        A = np.array([[random.randint(-10, 10) for _ in range(4)] for _ in range(4)])
        B = np.array([[random.randint(-10, 10) for _ in range(4)] for _ in range(4)])
        C_true = A @ B
        
        C_calc = np.zeros((4, 4))
        # C_ik = sum_r W[k*4+i, r] * (sum_j U[i*4+j, r] A_ij) * (sum_j V[j*4+k, r] B_jk)
        for i in range(4):
            for k in range(4):
                s = 0
                for r in range(49):
                    sum_u = 0
                    sum_v = 0
                    for j in range(4):
                        sum_u += U[i * 4 + j, r] * A[i, j]
                        sum_v += V[j * 4 + k, r] * B[j, k]
                    s += W[k * 4 + i, r] * sum_u * sum_v
                C_calc[i, k] = s
        
        if not np.allclose(C_true, C_calc):
            # Fallback to schoolbook if verification fails
            U_fb = np.zeros((16, 64))
            V_fb = np.zeros((16, 64))
            W_fb = np.zeros((16, 64))
            col = 0
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        U_fb[i * 4 + j, col] = 1
                        V_fb[j * 4 + k, col] = 1
                        W_fb[k * 4 + i, col] = 1
                        col += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
