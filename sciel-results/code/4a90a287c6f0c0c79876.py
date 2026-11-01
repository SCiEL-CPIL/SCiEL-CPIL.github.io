import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = 2, 3, 8

    # Build the target tensor
    T_exp = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1

    # Block decomposition: split p=8 into 4 blocks of size 2
    # For each block, compute 2x3 * 3x2
    # Split m=3 into m1=1 and m2=2
    # Part 1: 2x1 * 1x2 (rank 4)
    # Part 2: 2x2 * 2x2 (rank 7, Strassen)
    # Total per block: 11, total: 44

    rank = 44
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)

    col = 0

    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1

        # Part 1: A[:,0] * B[0, k0:k1] -> 2x1 * 1x2, rank 4
        # C[i, k] += A[i,0] * B[0, k]
        for i in range(n):
            for k in [k0, k1]:
                U[i*m + 0, col] = 1
                V[0*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1

        # Part 2: A[:,1:3] * B[1:3, k0:k1] -> 2x2 * 2x2, Strassen rank 7
        # A1 = A[:,1:3] = [[a01, a02], [a11, a12]]
        # B1 = B[1:3, k0:k1] = [[b10, b11], [b20, b21]]
        # Strassen products:
        # M1 = (a01 + a12) * (b10 + b21)
        # M2 = (a11 + a12) * b10
        # M3 = a01 * (b11 - b21)
        # M4 = a12 * (b20 - b10)
        # M5 = (a01 + a02) * b21
        # M6 = (a11 - a01) * (b10 + b11)
        # M7 = (a02 - a12) * (b20 + b21)
        #
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6

        # M1 = (a01 + a12) * (b10 + b21)
        U[0*m + 1, col] = 1
        U[1*m + 2, col] = 1
        V[1*p + k0, col] = 1
        V[2*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        W[k1*n + 1, col] = 1
        col += 1

        # M2 = (a11 + a12) * b10
        U[1*m + 1, col] = 1
        U[1*m + 2, col] = 1
        V[1*p + k0, col] = 1
        W[k0*n + 1, col] = 1
        W[k1*n + 1, col] = -1
        col += 1

        # M3 = a01 * (b11 - b21)
        U[0*m + 1, col] = 1
        V[1*p + k1, col] = 1
        V[2*p + k1, col] = -1
        W[k1*n + 0, col] = 1
        W[k1*n + 1, col] = 1
        col += 1

        # M4 = a12 * (b20 - b10)
        U[1*m + 2, col] = 1
        V[2*p + k0, col] = 1
        V[1*p + k0, col] = -1
        W[k0*n + 0, col] = 1
        W[k0*n + 1, col] = 1
        col += 1

        # M5 = (a01 + a02) * b21
        U[0*m + 1, col] = 1
        U[0*m + 2, col] = 1
        V[2*p + k1, col] = 1
        W[k0*n + 0, col] = -1
        W[k1*n + 0, col] = 1
        col += 1

        # M6 = (a11 - a01) * (b10 + b11)
        U[1*m + 1, col] = 1
        U[0*m + 1, col] = -1
        V[1*p + k0, col] = 1
        V[1*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1

        # M7 = (a02 - a12) * (b20 + b21)
        U[0*m + 2, col] = 1
        U[1*m + 2, col] = -1
        V[2*p + k0, col] = 1
        V[2*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def verify(U, V, W):
        r = U.shape[1]
        T_check = np.einsum('ad,bd,cd->abc', U, V, W)
        return np.array_equal(T_check, T_exp)

    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n*m, rank_schoolbook), dtype=np.int64)
        V_sb = np.zeros((m*p, rank_schoolbook), dtype=np.int64)
        W_sb = np.zeros((p*n, rank_schoolbook), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    U_sb[a_idx, col] = 1
                    V_sb[b_idx, col] = 1
                    W_sb[c_idx, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
