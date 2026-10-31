import numpy as np
import time

def construct():
    """
    Returns a rank-49 decomposition of the 4x4x4 matrix multiplication tensor
    via recursive Strassen (2x2 outer Strassen, each term expanded with 2x2 inner Strassen).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n, m, p = 4, 4, 4
    rank = 49  # 7 * 7: outer Strassen (7) x inner Strassen (7)

    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    # Strassen 2x2 coefficients
    # For 2x2 matrices: C = A * B
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # beta[s][I][J]: coefficient of A[I,J] in outer term s
    # gamma[s][J][K]: coefficient of B[J,K] in outer term s
    # delta[s][I][K]: coefficient of C[I,K] in outer term s
    beta = np.zeros((7, 2, 2), dtype=np.int64)
    gamma = np.zeros((7, 2, 2), dtype=np.int64)
    delta = np.zeros((7, 2, 2), dtype=np.int64)

    # s=0 (M1)
    beta[0, 0, 0] = 1; beta[0, 1, 1] = 1
    gamma[0, 0, 0] = 1; gamma[0, 1, 1] = 1
    delta[0, 0, 0] = 1; delta[0, 1, 1] = 1

    # s=1 (M2)
    beta[1, 1, 0] = 1; beta[1, 1, 1] = 1
    gamma[1, 0, 0] = 1
    delta[1, 1, 0] = 1; delta[1, 1, 1] = -1

    # s=2 (M3)
    beta[2, 0, 0] = 1
    gamma[2, 0, 1] = 1; gamma[2, 1, 1] = -1
    delta[2, 0, 1] = 1; delta[2, 1, 1] = 1

    # s=3 (M4)
    beta[3, 1, 1] = 1
    gamma[3, 1, 0] = 1; gamma[3, 0, 0] = -1
    delta[3, 0, 0] = 1; delta[3, 1, 0] = 1

    # s=4 (M5)
    beta[4, 0, 0] = 1; beta[4, 0, 1] = 1
    gamma[4, 1, 1] = 1
    delta[4, 0, 0] = -1; delta[4, 0, 1] = 1

    # s=5 (M6)
    beta[5, 1, 0] = 1; beta[5, 0, 0] = -1
    gamma[5, 0, 0] = 1; gamma[5, 0, 1] = 1
    delta[5, 1, 1] = 1

    # s=6 (M7)
    beta[6, 0, 1] = 1; beta[6, 1, 1] = -1
    gamma[6, 1, 0] = 1; gamma[6, 1, 1] = 1
    delta[6, 0, 0] = 1

    # Inner Strassen 2x2 (same structure)
    # U2[i,j,t]: coefficient of A_local[i,j] in inner term t
    # V2[j,k,t]: coefficient of B_local[j,k] in inner term t
    # W2[k,i,t]: coefficient of C_local[k,i] in inner term t
    U2 = np.zeros((4, 7), dtype=np.int64)
    V2 = np.zeros((4, 7), dtype=np.int64)
    W2 = np.zeros((4, 7), dtype=np.int64)

    # t=0 (M1 inner): X=A00+A11, Y=B00+B11, C00+=M1, C11+=M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # t=1 (M2 inner): X=A10+A11, Y=B00, C10+=M2, C11-=M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1

    # t=2 (M3 inner): X=A00, Y=B01-B11, C01+=M3, C11+=M3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1

    # t=3 (M4 inner): X=A11, Y=B10-B00, C00+=M4, C10+=M4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1

    # t=4 (M5 inner): X=A00+A01, Y=B11, C00-=M5, C01+=M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1

    # t=5 (M6 inner): X=A10-A00, Y=B00+B01, C11+=M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # t=6 (M7 inner): X=A01-A11, Y=B10+B11, C00+=M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Build full U, V, W by combining outer and inner Strassen
    for s in range(7):
        for t in range(7):
            c = 7 * s + t

            # U: rows are (i,j) flattened as i*4+j
            # U[i*4+j, c] = beta[s][I][J] * U2[i'*2+j', t]
            # where i = 2*I + i', j = 2*J + j'
            for I in range(2):
                for J in range(2):
                    bs = beta[s, I, J]
                    if bs == 0:
                        continue
                    for ip in range(2):
                        for jp in range(2):
                            u2_val = U2[ip * 2 + jp, t]
                            if u2_val == 0:
                                continue
                            i = 2 * I + ip
                            j = 2 * J + jp
                            U[i * 4 + j, c] = bs * u2_val

            # V: rows are (j,k) flattened as j*4+k
            # V[j*4+k, c] = gamma[s][J][K] * V2[j'*2+k', t]
            # where j = 2*J + j', k = 2*K + k'
            for J in range(2):
                for K in range(2):
                    gs = gamma[s, J, K]
                    if gs == 0:
                        continue
                    for jp in range(2):
                        for kp in range(2):
                            v2_val = V2[jp * 2 + kp, t]
                            if v2_val == 0:
                                continue
                            j = 2 * J + jp
                            k = 2 * K + kp
                            V[j * 4 + k, c] = gs * v2_val

            # W: rows are (k,i) flattened as k*4+i (trace convention W[k*n+i, r])
            # W[k*4+i, c] = delta[s][I][K] * W2[k'*2+i', t]
            # where i = 2*I + i', k = 2*K + k'
            for I in range(2):
                for K in range(2):
                    ds = delta[s, I, K]
                    if ds == 0:
                        continue
                    for ip in range(2):
                        for kp in range(2):
                            w2_val = W2[kp * 2 + ip, t]
                            if w2_val == 0:
                                continue
                            i = 2 * I + ip
                            k = 2 * K + kp
                            W[k * 4 + i, c] = ds * w2_val

            if time.time() > deadline:
                # Return what we have so far
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verify using numpy for speed
    # Check: sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] == 1 for all i,j,k
    # Reshape: U is (16, 49), V is (16, 49), W is (16, 49)
    # For each (i,j,k): sum over r of U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    
    # Vectorized check:
    # Create tensors of shape (4,4,4,49) for U, V, W
    U_t = U.reshape(4, 4, 49)  # U[i,j,r]
    V_t = V.reshape(4, 4, 49)  # V[j,k,r]
    W_t = W.reshape(4, 4, 49)  # W[k,i,r]
    
    # Compute sum over r for all (i,j,k)
    # result[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # This is a batched einsum
    result = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    
    if not np.all(result == 1):
        # Fallback: schoolbook rank 64
        rank_fb = 64
        U_fb = np.zeros((16, rank_fb), dtype=np.int64)
        V_fb = np.zeros((16, rank_fb), dtype=np.int64)
        W_fb = np.zeros((16, rank_fb), dtype=np.int64)
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
