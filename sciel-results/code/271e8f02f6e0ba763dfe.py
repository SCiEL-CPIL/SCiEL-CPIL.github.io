import time
import sys
import numpy as np

DIMENSIONS = [2, 2, 3]

def construct():
    deadline = time.time() + 148.0
    n, m, p = DIMENSIONS  # 2, 2, 3

    rank = 10
    U = np.zeros((n * m, rank))  # 4 x 10
    V = np.zeros((m * p, rank))  # 6 x 10
    W = np.zeros((p * n, rank))  # 6 x 10

    # Indices:
    # U: i*m+j  (i in 0..1, j in 0..1)
    # V: j*p+k  (j in 0..1, k in 0..2)
    # W: k*n+i  (k in 0..2, i in 0..1)

    # --- Strassen 7 terms for k=0,1 ---
    # M1 = (A00+A11)(B00+B11) -> C00+C11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[4, 0] = 1

    # M2 = (A10+A11)B00 -> C10
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[3, 1] = 1

    # M3 = A00(B01-B11) -> C01-C11
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[1, 2] = 1; W[4, 2] = -1

    # M4 = A11(B10-B00) -> C10-C00
    U[3, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[3, 3] = 1; W[0, 3] = -1

    # M5 = (A00+A01)B11 -> C01+C11
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[1, 4] = 1; W[4, 4] = 1

    # M6 = (A10-A00)(B00+B01) -> C10+C11
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1; W[4, 5] = 1

    # M7 = (A01-A11)(B10+B11) -> C00+C11
    U[1, 6] = 1; U[3, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1; W[4, 6] = 1

    # --- 3 terms for k=2 ---
    # Term 7: A00 B02 -> C02
    U[0, 7] = 1
    V[2, 7] = 1
    W[2, 7] = 1

    # Term 8: A10 B02 -> C12
    U[2, 8] = 1
    V[2, 8] = 1
    W[5, 8] = 1

    # Term 9: (A01+A11) B12 -> C02+C12
    U[1, 9] = 1; U[3, 9] = 1
    V[5, 9] = 1
    W[2, 9] = 1; W[5, 9] = 1

    # Verify
    expected = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                expected.add((a, b, d))

    for a in range(n * m):
        for b in range(m * p):
            for d in range(p * n):
                total = 0
                for r in range(rank):
                    total += U[a, r] * V[b, r] * W[d, r]
                expected_val = 1 if (a, b, d) in expected else 0
                if total != expected_val:
                    sys.stderr.write(f"Verification failed\n")
                    # Fallback to schoolbook rank 12
                    rank_f = 12
                    Uf = np.zeros((n * m, rank_f))
                    Vf = np.zeros((m * p, rank_f))
                    Wf = np.zeros((p * n, rank_f))
                    col = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                Uf[i * m + j, col] = 1
                                Vf[j * p + k, col] = 1
                                Wf[k * n + i, col] = 1
                                col += 1
                    return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
