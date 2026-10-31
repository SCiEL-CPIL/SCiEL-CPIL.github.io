import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin

    n, m, p = 4, 4, 4
    rank = 49

    # Initialize factor matrices as NumPy arrays for efficiency
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen 2x2 decomposition coefficients
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # M0 = (A00 + A11) * (B00 + B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # M1 = (A10 + A11) * B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1

    # M2 = A00 * (B01 - B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1

    # M3 = A11 * (B10 - B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1

    # M4 = (A00 + A01) * B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1

    # M5 = (A10 - A00) * (B00 + B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # M6 = (A01 - A11) * (B10 + B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Construct 4x4 decomposition using Kronecker product logic
    for s in range(7):
        for t in range(7):
            rank_idx = s * 7 + t
            
            # Update U
            # U_flat[i*4+j, rank_idx] = U_block[I*2+J, s] * U_inner[ii*2+jj, t]
            for i in range(4):
                for j in range(4):
                    I = i // 2
                    J = j // 2
                    ii = i % 2
                    jj = j % 2
                    coeff = U2[I * 2 + J, s] * U2[ii * 2 + jj, t]
                    if coeff != 0:
                        U[i * 4 + j, rank_idx] = coeff

            # Update V
            # V_flat[k*4+l, rank_idx] = V_block[K*2+L, s] * V_inner[kk*2+ll, t]
            for k in range(4):
                for l in range(4):
                    K = k // 2
                    L = l // 2
                    kk = k % 2
                    ll = l % 2
                    coeff = V2[K * 2 + L, s] * V2[kk * 2 + ll, t]
                    if coeff != 0:
                        V[k * 4 + l, rank_idx] = coeff

            # Update W
            # W_flat[j*4+i, rank_idx] = W_block[I*2+J, s] * W_inner[ii*2+jj, t]
            # Note: C[j, i] corresponds to W row j*4+i
            for j in range(4):
                for i in range(4):
                    J = j // 2
                    I = i // 2
                    jj = j % 2
                    ii = i % 2
                    coeff = W2[I * 2 + J, s] * W2[ii * 2 + jj, t]
                    if coeff != 0:
                        W[j * 4 + i, rank_idx] = coeff

            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
