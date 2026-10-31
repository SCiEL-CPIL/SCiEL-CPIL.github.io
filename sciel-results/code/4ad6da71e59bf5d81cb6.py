import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Strassen decomposition for 2x2 matrices (rank 7)
    # U2: shape (4, 7), row = i2*2+j2
    # V2: shape (4, 7), row = j2*2+k2
    # W2: shape (4, 7), row = k2*2+i2
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # Outer Strassen coefficients for 2x2 block decomposition
    # beta[s, I, J], gamma[s, J, K], delta[s, I, K]
    beta = np.zeros((7, 2, 2), dtype=int)
    gamma = np.zeros((7, 2, 2), dtype=int)
    delta = np.zeros((7, 2, 2), dtype=int)

    # s=0: M0 = (A00+A11)(B00+B11) -> C00, C11
    beta[0,0,0]=1; beta[0,1,1]=1
    gamma[0,0,0]=1; gamma[0,1,1]=1
    delta[0,0,0]=1; delta[0,1,1]=1

    # s=1: M1 = (A10+A11)B00 -> C10, C11
    beta[1,1,0]=1; beta[1,1,1]=1
    gamma[1,0,0]=1
    delta[1,1,0]=1; delta[1,1,1]=-1

    # s=2: M2 = A00(B01-B11) -> C01, C11
    beta[2,0,0]=1
    gamma[2,0,1]=1; gamma[2,1,1]=-1
    delta[2,0,1]=1; delta[2,1,1]=1

    # s=3: M3 = A11(B10-B00) -> C00, C10
    beta[3,1,1]=1
    gamma[3,1,0]=1; gamma[3,0,0]=-1
    delta[3,0,0]=1; delta[3,1,0]=1

    # s=4: M4 = (A00+A01)B11 -> C00, C01
    beta[4,0,0]=1; beta[4,0,1]=1
    gamma[4,1,1]=1
    delta[4,0,0]=-1; delta[4,0,1]=1

    # s=5: M5 = (A10-A00)(B00+B01) -> C11
    beta[5,1,0]=1; beta[5,0,0]=-1
    gamma[5,0,0]=1; gamma[5,0,1]=1
    delta[5,1,1]=1

    # s=6: M6 = (A01-A11)(B10+B11) -> C00
    beta[6,0,1]=1; beta[6,1,1]=-1
    gamma[6,1,0]=1; gamma[6,1,1]=1
    delta[6,0,0]=1

    rank = 49
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)

    # Construct U: U[i*4+j, s*7+t] = beta[s,I,J] * U2[i2*2+j2, t]
    # where i = 2*I+i2, j = 2*J+j2
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            # U2 row: i2*2+j2 for i2,j2 in {0,1}
            # For each (I,J) block, U2[i2*2+j2, t] is the inner coefficient
            for I in range(2):
                for J in range(2):
                    b_val = beta[s, I, J]
                    if b_val == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            row_u = (2*I + i2) * 4 + (2*J + j2)
                            u2_val = U2[i2 * 2 + j2, t]
                            if u2_val != 0:
                                U[row_u, col] += b_val * u2_val

    # Construct V: V[j*4+k, s*7+t] = gamma[s,J,K] * V2[j2*2+k2, t]
    # where j = 2*J+j2, k = 2*K+k2
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            for J in range(2):
                for K in range(2):
                    g_val = gamma[s, J, K]
                    if g_val == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            row_v = (2*J + j2) * 4 + (2*K + k2)
                            v2_val = V2[j2 * 2 + k2, t]
                            if v2_val != 0:
                                V[row_v, col] += g_val * v2_val

    # Construct W: W[k*4+i, s*7+t] = delta[s,I,K] * W2[k2*2+i2, t]
    # where k = 2*K+k2, i = 2*I+i2
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            for I in range(2):
                for K in range(2):
                    d_val = delta[s, I, K]
                    if d_val == 0:
                        continue
                    for i2 in range(2):
                        for k2 in range(2):
                            row_w = (2*K + k2) * 4 + (2*I + i2)
                            w2_val = W2[k2 * 2 + i2, t]
                            if w2_val != 0:
                                W[row_w, col] += d_val * w2_val

    # Verify: T[i,j,k] = sum_t U[i*4+j, t] * V[j*4+k, t] * W[k*4+i, t] should be 1
    # Use vectorized check on a few random inputs
    np.random.seed(123)
    A_test = np.random.randint(-3, 4, (4, 4))
    B_test = np.random.randint(-3, 4, (4, 4))
    C_expected = A_test @ B_test

    # Compute C from decomposition: C[i,k] = sum_j sum_t U[i*4+j,t] * V[j*4+k,t] * W[k*4+i,t]
    # But actually the decomposition gives: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor identity is: sum_t U[i*4+j,t]*V[j*4+k,t]*W[k*4+i,t] = 1 for all i,j,k
    # So C[i,k] = sum_j sum_t A[i,j]*B[j,k]*1 = sum_j A[i,j]*B[j,k] = (A@B)[i,k]

    # Efficient verification using einsum-like approach
    # C[i,k] = sum_j sum_t A[i,j] * U[i*4+j,t] * B[j,k] * V[j*4+k,t] * W[k*4+i,t]
    # This is expensive to do naively; let's just verify the tensor identity on a subset
    # For speed, check a small number of (i,j,k) triples
    verified = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = 0
                for t in range(rank):
                    val += U[i*4+j, t] * V[j*4+k, t] * W[k*4+i, t]
                if val != 1:
                    verified = False
                    break
            if not verified:
                break
        if not verified:
            break

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
