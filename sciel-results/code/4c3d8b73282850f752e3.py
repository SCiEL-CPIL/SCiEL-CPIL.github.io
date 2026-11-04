import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 120.0

    # Strassen 2x2x2 (Rank 7)
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)

    # Verify 2x2x2
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = sum(U2[i, t] * V2[j, t] * W2[k, t] for t in range(7))
                expected = 1 if (i == k) else 0  # No, this isn't right for 2x2x2
    # Actually for matrix multiplication 2x2x2: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 always for the multiplication tensor
    # Check: sum_t U2[i,t]*V2[j,t]*W2[k,t] should equal 1 for all i,j,k
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = sum(U2[i, t] * V2[j, t] * W2[k, t] for t in range(7))
                if abs(val - 1) > 1e-10:
                    print(f"2x2x2 fail: i={i},j={j},k={k},val={val}")

    # Build 4x4x4 (Rank 49) using Strassen recursion
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))

    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            U4[i*4+j][c] = U2[i1*2+j1][t] * U2[i2*2+j2][s]

    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            V4[j*4+k][c] = V2[j1*2+k1][t] * V2[j2*2+k2][s]

    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    for t in range(7):
                        for s in range(7):
                            c = t*7 + s
                            W4[k*4+i][c] = W2[k1*2+i1][t] * W2[k2*2+i2][s]

    # Verify 4x4x4
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = sum(U4[i*4+j, t] * V4[j*4+k, t] * W4[k*4+i, t] for t in range(49))
                if abs(val - 1) > 1e-10:
                    print(f"4x4x4 fail: i={i},j={j},k={k},val={val}")

    # Build 4x4x8 (Rank 98)
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))

    U[:, :49] = U4
    U[:, 49:] = U4

    for j in range(4):
        for k in range(4):
            V[j*8+k, :49] = V4[j*4+k]
            V[j*8+k+4, 49:] = V4[j*4+k]

    for k in range(4):
        for i in range(4):
            W[k*4+i, :49] = W4[k*4+i]
            W[(k+4)*4+i, 49:] = W4[k*4+i]

    # Verify 4x4x8
    for i in range(4):
        for j in range(4):
            for k in range(8):
                val = sum(U[i*4+j, t] * V[j*8+k, t] * W[k*4+i, t] for t in range(98))
                if abs(val - 1) > 1e-10:
                    print(f"4x4x8 fail: i={i},j={j},k={k},val={val}")

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
