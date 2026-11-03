import time
import numpy as np

def construct():
    t0 = time.time()
    limit = t0 + 145.0

    N, M, P = 2, 4, 4
    R = 28

    U = np.zeros((N * M, R), dtype=np.int64)
    V = np.zeros((M * P, R), dtype=np.int64)
    W = np.zeros((P * N, R), dtype=np.int64)

    def idx_a(r, c):
        return r * M + c

    def idx_b(r, c):
        return r * P + c

    def idx_c(row, col):
        return col * N + row

    def strassen_add(ar0, ac0, br0, bc0, cr0, cc0, off):
        a11, a12, a21, a22 = idx_a(ar0, ac0), idx_a(ar0, ac0+1), idx_a(ar0+1, ac0), idx_a(ar0+1, ac0+1)
        b11, b12, b21, b22 = idx_b(br0, bc0), idx_b(br0, bc0+1), idx_b(br0+1, bc0), idx_b(br0+1, bc0+1)
        c11, c12, c21, c22 = idx_c(cr0, cc0), idx_c(cr0, cc0+1), idx_c(cr0+1, cc0), idx_c(cr0+1, cc0+1)

        # M1 = (a11+a22)(b11+b22) -> c11 += , c22 +=
        k = off
        U[a11, k] = 1; U[a22, k] = 1
        V[b11, k] = 1; V[b22, k] = 1
        W[c11, k] = 1; W[c22, k] = 1

        # M2 = (a21+a22)*b11 -> c21 += , c22 -=
        k = off + 1
        U[a21, k] = 1; U[a22, k] = 1
        V[b11, k] = 1
        W[c21, k] = 1; W[c22, k] = -1

        # M3 = a11*(b12-b22) -> c12 += , c22 +=
        k = off + 2
        U[a11, k] = 1
        V[b12, k] = 1; V[b22, k] = -1
        W[c12, k] = 1; W[c22, k] = 1

        # M4 = a22*(b21-b11) -> c11 += , c21 +=
        k = off + 3
        U[a22, k] = 1
        V[b21, k] = 1; V[b11, k] = -1
        W[c11, k] = 1; W[c21, k] = 1

        # M5 = (a11+a12)*b22 -> c11 -= , c12 +=
        k = off + 4
        U[a11, k] = 1; U[a12, k] = 1
        V[b22, k] = 1
        W[c11, k] = -1; W[c12, k] = 1

        # M6 = (a21-a11)*(b11+b12) -> c22 +=
        k = off + 5
        U[a21, k] = 1; U[a11, k] = -1
        V[b11, k] = 1; V[b12, k] = 1
        W[c22, k] = 1

        # M7 = (a12-a22)*(b21+b22) -> c11 +=
        k = off + 6
        U[a12, k] = 1; U[a22, k] = -1
        V[b21, k] = 1; V[b22, k] = 1
        W[c11, k] = 1

        return off + 7

    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    off = strassen_add(0, 0, 0, 0, 0, 0, 0)
    # Block 2: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    off = strassen_add(0, 2, 2, 0, 0, 0, off)
    # Block 3: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    off = strassen_add(0, 0, 0, 2, 0, 2, off)
    # Block 4: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    off = strassen_add(0, 2, 2, 2, 0, 2, off)

    assert off == R

    if time.time() > limit:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
