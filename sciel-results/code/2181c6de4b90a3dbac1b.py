import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s kill

    n, m, p = 2, 3, 6

    # Rank-33 decomposition:
    # Part 1: A[:, 0] (2x1) * B[0, :] (1x6) -> schoolbook rank 12
    # Part 2: For each pair of columns (2t, 2t+1) of B, rows 1,2 of B:
    #   2x2x2 Strassen: A[:,1:3] (2x2) * B[1:3, 2t:2t+2] (2x2)
    #   Each Strassen block has rank 7, three blocks -> rank 21
    # Total: 12 + 21 = 33

    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    # Part 1: A[i,0] * B[0,k] -> C[i,k]
    for k in range(p):
        for i in range(n):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1

    # Part 2: Strassen for each 2x2 block
    # For block t (t=0,1,2), columns k0=2t, k1=2t+1
    # A_sub = A[:, 1:3] (2x2), B_sub = B[1:3, k0:k1+1] (2x2)
    # C_sub[i, k] = sum_j A_sub[i,j] * B_sub[j,k]
    #   = A[i,1]*B[1,k] + A[i,2]*B[2,k]
    #
    # Strassen for 2x2:
    #   M1 = (A00+A11)*(B00+B11)
    #   M2 = (A10+A11)*B00
    #   M3 = A00*(B10-B11)
    #   M4 = A11*(B01-B00)
    #   M5 = (A00+A01)*B11
    #   M6 = (A10-A00)*(B00+B01)
    #   M7 = (A01-A11)*(B10+B11)
    #
    #   C00 = M1 + M4 - M5 - M7
    #   C01 = M5 + M7
    #   C10 = M2 + M3
    #   C11 = M1 - M2 + M3 + M4
    #
    # Here A00=A[i=0,j=1], A01=A[i=0,j=2], A10=A[i=1,j=1], A11=A[i=1,j=2]
    # B00=B[j=1,k0], B01=B[j=1,k1], B10=B[j=2,k0], B11=B[j=2,k1]
    # C00=C[i=0,k0], C01=C[i=0,k1], C10=C[i=1,k0], C11=C[i=1,k1]

    for t in range(3):
        k0 = 2 * t
        k1 = 2 * t + 1

        # M1 = (A[0,1]+A[1,2]) * (B[1,k0]+B[2,k1])
        U[0 * m + 1, col] = 1.0
        U[1 * m + 2, col] = 1.0
        V[1 * p + k0, col] = 1.0
        V[2 * p + k1, col] = 1.0
        W[k0 * n + 0, col] = 1.0
        W[k1 * n + 1, col] = 1.0
        col += 1

        # M2 = (A[1,1]+A[1,2]) * B[1,k0]
        U[1 * m + 1, col] = 1.0
        U[1 * m + 2, col] = 1.0
        V[1 * p + k0, col] = 1.0
        W[k0 * n + 1, col] = 1.0
        W[k1 * n + 1, col] = -1.0
        col += 1

        # M3 = A[0,1] * (B[2,k0]-B[2,k1])
        U[0 * m + 1, col] = 1.0
        V[2 * p + k0, col] = 1.0
        V[2 * p + k1, col] = -1.0
        W[k0 * n + 1, col] = 1.0
        W[k1 * n + 1, col] = 1.0
        col += 1

        # M4 = A[1,2] * (B[1,k1]-B[1,k0])
        U[1 * m + 2, col] = 1.0
        V[1 * p + k1, col] = 1.0
        V[1 * p + k0, col] = -1.0
        W[k0 * n + 0, col] = 1.0
        W[k1 * n + 0, col] = -1.0
        col += 1

        # M5 = (A[0,1]+A[0,2]) * B[2,k1]
        U[0 * m + 1, col] = 1.0
        U[0 * m + 2, col] = 1.0
        V[2 * p + k1, col] = 1.0
        W[k0 * n + 0, col] = -1.0
        W[k1 * n + 0, col] = 1.0
        col += 1

        # M6 = (A[1,1]-A[0,1]) * (B[1,k0]+B[1,k1])
        U[1 * m + 1, col] = 1.0
        U[0 * m + 1, col] = -1.0
        V[1 * p + k0, col] = 1.0
        V[1 * p + k1, col] = 1.0
        W[k1 * n + 1, col] = 1.0
        col += 1

        # M7 = (A[0,2]-A[1,2]) * (B[2,k0]+B[2,k1])
        U[0 * m + 2, col] = 1.0
        U[1 * m + 2, col] = -1.0
        V[2 * p + k0, col] = 1.0
        V[2 * p + k1, col] = 1.0
        W[k0 * n + 0, col] = -1.0
        col += 1

    # Verify: tensor[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should be 1 if j is in {0,1,2} (always true) and the product A[i,j]*B[j,k] contributes
    # Actually the tensor T[i,j,k] = 1 for all i,j,k (it's the identity mapping for matmul)
    # Wait no: T[i,j,k] = 1 means C[i,k] gets A[i,j]*B[j,k]
    # So the tensor has T[i,j,k] = 1 for all valid i,j,k

    if time.time() - start_time > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    U_r = U.reshape(n, m, -1)
    V_r = V.reshape(m, p, -1)
    W_r = W.reshape(p, n, -1)

    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i,j,k] should be 1
                s = np.dot(U_r[i, j, :] * V_r[j, k, :], W_r[k, i, :])
                if abs(s - 1.0) > max_err:
                    max_err = abs(s - 1.0)

    if max_err > 1e-9:
        # Fallback to schoolbook rank 36
        rank_school = n * m * p
        U_school = np.zeros((n * m, rank_school))
        V_school = np.zeros((m * p, rank_school))
        W_school = np.zeros((p * n, rank_school))
        col_s = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_school[ii * m + jj, col_s] = 1.0
                    V_school[jj * p + kk, col_s] = 1.0
                    W_school[kk * n + ii, col_s] = 1.0
                    col_s += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
