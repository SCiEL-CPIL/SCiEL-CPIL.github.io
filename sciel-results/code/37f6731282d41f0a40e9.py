import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 110

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def schoolbook_3x3x3():
        rank = 27
        U = np.zeros((9, rank))
        V = np.zeros((9, rank))
        W = np.zeros((9, rank))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3 + j, col] = 1.0
                    V[j*3 + k, col] = 1.0
                    W[k*3 + i, col] = 1.0
                    col += 1
        return U, V, W

    def schoolbook_3x1x3():
        rank = 3
        U = np.zeros((9, rank))
        V = np.zeros((3, rank))
        W = np.zeros((3, rank))
        col = 0
        for i in range(3):
            for k in range(3):
                U[i*3 + 0, col] = 1.0
                V[0*3 + k, col] = 1.0
                W[k*3 + i, col] = 1.0
                col += 1
        return U, V, W

    U_total = []
    V_total = []
    W_total = []
    
    # Part 1: A * B11 (3x3 * 3x3)
    U1, V1, W1 = schoolbook_3x3x3()
    V1_full = np.zeros((24, 27))
    for r in range(9):
        for c in range(27):
            if V1[r, c] != 0:
                V1_full[r, c] = V1[r, c]
    W1_full = np.zeros((24, 27))
    for r in range(9):
        for c in range(27):
            if W1[r, c] != 0:
                W1_full[r, c] = W1[r, c]
    U1_full = np.zeros((9, 27))
    U1_full[:] = U1
    
    U_total.append(U1_full)
    V_total.append(V1_full)
    W_total.append(W1_full)
    
    # Part 2: A * b12 (3x3 * 3x1)
    U2, V2, W2 = schoolbook_3x1x3()
    V2_full = np.zeros((24, 3))
    for r in range(3):
        for c in range(3):
            if V2[r, c] != 0:
                V2_full[3 + r, c] = V2[r, c]
    W2_full = np.zeros((24, 3))
    for r in range(3):
        for c in range(3):
            if W2[r, c] != 0:
                W2_full[3 + r, c] = W2[r, c]
    U2_full = np.zeros((9, 3))
    U2_full[:] = U2
    
    U_total.append(U2_full)
    V_total.append(V2_full)
    W_total.append(W2_full)
    
    # Part 3: A * B21 (3x3 * 3x3)
    U3, V3, W3 = schoolbook_3x3x3()
    V3_full = np.zeros((24, 27))
    for r in range(9):
        for c in range(27):
            if V3[r, c] != 0:
                V3_full[4 + r, c] = V3[r, c]
    W3_full = np.zeros((24, 27))
    for r in range(9):
        for c in range(27):
            if W3[r, c] != 0:
                W3_full[4 + r, c] = W3[r, c]
    U3_full = np.zeros((9, 27))
    U3_full[:] = U3
    
    U_total.append(U3_full)
    V_total.append(V3_full)
    W_total.append(W3_full)
    
    # Part 4: A * b22 (3x3 * 3x1)
    U4, V4, W4 = schoolbook_3x1x3()
    V4_full = np.zeros((24, 3))
    for r in range(3):
        for c in range(3):
            if V4[r, c] != 0:
                V4_full[7 + r, c] = V4[r, c]
    W4_full = np.zeros((24, 3))
    for r in range(3):
        for c in range(3):
            if W4[r, c] != 0:
                W4_full[7 + r, c] = W4[r, c]
    U4_full = np.zeros((9, 3))
    U4_full[:] = U4
    
    U_total.append(U4_full)
    V_total.append(V4_full)
    W_total.append(W4_full)
    
    U_final = np.hstack(U_total)
    V_final = np.hstack(V_total)
    W_final = np.hstack(W_total)
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
