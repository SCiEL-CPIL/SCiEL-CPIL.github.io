import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 2x2 Strassen Decomposition
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # t=0: M1 = (a+d)(e+h)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # t=1: M2 = (c+d)e
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # t=2: M3 = a(f-h)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # t=3: M4 = d(g-e)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # t=4: M5 = (a+b)h
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # t=5: M6 = (c-a)(e+f)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # t=6: M7 = (b-d)(g+h)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # 3x3 Schoolbook Decomposition
    # U3: 9x27, V3: 9x27, W3: 9x27
    U3 = np.zeros((9, 27), dtype=int)
    V3 = np.zeros((9, 27), dtype=int)
    W3 = np.zeros((9, 27), dtype=int)
    
    for i3 in range(3):
        for j3 in range(3):
            for k3 in range(3):
                t3 = i3 * 9 + j3 * 3 + k3
                r_u3 = i3 * 3 + j3
                r_v3 = j3 * 3 + k3
                r_w3 = k3 * 3 + i3
                U3[r_u3, t3] = 1
                V3[r_v3, t3] = 1
                W3[r_w3, t3] = 1

    # Compute Kronecker Products
    U_kron = np.kron(U3, U2)
    V_kron = np.kron(V3, V2)
    W_kron = np.kron(W3, W2)
    
    # Permute rows to match standard 6x6 layout
    # Standard index: S = i * 6 + j
    # Kronecker index: K = r3 * 4 + r2
    # Mapping:
    # i = i3 * 2 + i2, j = j3 * 2 + j2
    # S = i3*12 + i2*6 + j3*2 + j2
    # K = (i3*3 + j3)*4 + (i2*2 + j2) = i3*12 + j3*4 + i2*2 + j2
    
    U_new = np.zeros((36, 189), dtype=int)
    V_new = np.zeros((36, 189), dtype=int)
    W_new = np.zeros((36, 189), dtype=int)
    
    for i3 in range(3):
        for i2 in range(2):
            i = i3 * 2 + i2
            for j3 in range(3):
                for j2 in range(2):
                    j = j3 * 2 + j2
                    s_u = i * 6 + j
                    r_u3 = i3 * 3 + j3
                    r_u2 = i2 * 2 + j2
                    k_u = r_u3 * 4 + r_u2
                    U_new[s_u, :] = U_kron[k_u, :]
                    
    for j3 in range(3):
        for j2 in range(2):
            j = j3 * 2 + j2
            for k3 in range(3):
                for k2 in range(2):
                    k = k3 * 2 + k2
                    s_v = j * 6 + k
                    r_v3 = j3 * 3 + k3
                    r_v2 = j2 * 2 + k2
                    k_v = r_v3 * 4 + r_v2
                    V_new[s_v, :] = V_kron[k_v, :]
                    
    for k3 in range(3):
        for k2 in range(2):
            k = k3 * 2 + k2
            for i3 in range(3):
                for i2 in range(2):
                    i = i3 * 2 + i2
                    s_w = k * 6 + i
                    r_w3 = k3 * 3 + i3
                    r_w2 = k2 * 2 + i2
                    k_w = r_w3 * 4 + r_w2
                    W_new[s_w, :] = W_kron[k_w, :]

    return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
# EVOLVE-BLOCK-END
