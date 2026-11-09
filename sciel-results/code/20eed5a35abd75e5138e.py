import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS  # n=2, m=3, p=10

    def verify(U, V, W, n, m, p):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(r))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Try Strassen-based rank-55 decomposition
    rank_55 = 55
    U55 = np.zeros((n * m, rank_55))
    V55 = np.zeros((m * p, rank_55))
    W55 = np.zeros((p * n, rank_55))

    c_idx = 0
    for col_pair in range(5):
        k0 = col_pair * 2
        k1 = col_pair * 2 + 1

        # Strassen 2x2x2 for A[0:2,0:2] * B[0:2,k0:k1+1]
        # 7 terms
        # M1 = (A00+A11)*(B00+B11)
        U55[0, c_idx] = 1; U55[4, c_idx] = 1
        V55[k0, c_idx] = 1; V55[10+k1, c_idx] = 1
        W55[k0*2+0, c_idx] = 1; W55[k1*2+1, c_idx] = 1
        c_idx += 1

        # M2 = (A10+A11)*B00
        U55[3, c_idx] = 1; U55[4, c_idx] = 1
        V55[k0, c_idx] = 1
        W55[k0*2+1, c_idx] = 1
        c_idx += 1

        # M3 = A00*(B01-B11)
        U55[0, c_idx] = 1
        V55[k1, c_idx] = 1; V55[10+k1, c_idx] = -1
        W55[k1*2+0, c_idx] = 1; W55[k1*2+1, c_idx] = -1
        c_idx += 1

        # M4 = A11*(B10-B00)
        U55[4, c_idx] = 1
        V55[10+k0, c_idx] = 1; V55[k0, c_idx] = -1
        W55[k0*2+0, c_idx] = -1; W55[k0*2+1, c_idx] = 1
        c_idx += 1

        # M5 = (A00+A01)*B11
        U55[0, c_idx] = 1; U55[1, c_idx] = 1
        V55[10+k1, c_idx] = 1
        W55[k1*2+0, c_idx] = 1; W55[k1*2+1, c_idx] = 1
        c_idx += 1

        # M6 = (A10-A00)*(B00+B01)
        U55[3, c_idx] = 1; U55[0, c_idx] = -1
        V55[k0, c_idx] = 1; V55[k1, c_idx] = 1
        W55[k0*2+0, c_idx] = 1; W55[k0*2+1, c_idx] = -1
        c_idx += 1

        # M7 = (A01-A11)*(B10+B11)
        U55[1, c_idx] = 1; U55[4, c_idx] = -1
        V55[10+k0, c_idx] = 1; V55[10+k1, c_idx] = 1
        W55[k0*2+0, c_idx] = -1; W55[k1*2+0, c_idx] = 1
        c_idx += 1

        # 4 standard terms for A[:,2] * B[2,:]
        U55[2, c_idx] = 1; V55[20+k0, c_idx] = 1; W55[k0*2+0, c_idx] = 1; c_idx += 1
        U55[2, c_idx] = 1; V55[20+k1, c_idx] = 1; W55[k1*2+0, c_idx] = 1; c_idx += 1
        U55[5, c_idx] = 1; V55[20+k0, c_idx] = 1; W55[k0*2+1, c_idx] = 1; c_idx += 1
        U55[5, c_idx] = 1; V55[20+k1, c_idx] = 1; W55[k1*2+1, c_idx] = 1; c_idx += 1

    if verify(U55, V55, W55, n, m, p):
        best_U, best_V, best_W = U55, V55, W55
    else:
        # Fallback to trivial rank-60
        rank60 = n * m * p
        U60 = np.zeros((n * m, rank60))
        V60 = np.zeros((m * p, rank60))
        W60 = np.zeros((p * n, rank60))
        for c in range(rank60):
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U60[i * m + j, c] = 1
            V60[j * p + k, c] = 1
            W60[k * n + i, c] = 1
        best_U, best_V, best_W = U60, V60, W60

    def to_exact(U_mat):
        U_list = []
        for row in U_mat:
            r = []
            for val in row:
                v = round(val * 2) / 2.0
                if abs(v - round(v)) < 1e-9:
                    r.append(int(round(v)))
                else:
                    r.append(v)
            U_list.append(r)
        return U_list

    return {"U": to_exact(best_U), "V": to_exact(best_V), "W": to_exact(best_W)}
