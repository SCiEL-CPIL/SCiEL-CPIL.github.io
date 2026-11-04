import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = 3, 4, 5

    # Schoolbook decomposition: rank = n*m*p = 60
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    # Verify correctness with multiple random tests
    np.random.seed(12345)
    for trial in range(5):
        A_test = np.random.randint(-50, 50, (n, m))
        B_test = np.random.randint(-50, 50, (m, p))
        C_expected = A_test @ B_test

        C_recomputed = np.zeros((n, p), dtype=np.int64)
        for c in range(rank):
            ua = np.dot(U[:, c], A_test.ravel())
            vb = np.dot(V[:, c], B_test.ravel())
            Wc = W[:, c].reshape(p, n).T
            C_recomputed += Wc * (ua * vb)

        if not np.array_equal(C_recomputed, C_expected):
            # Should never happen for schoolbook, but if it does, rebuild
            U = np.zeros((n * m, rank), dtype=np.int64)
            V = np.zeros((m * p, rank), dtype=np.int64)
            W = np.zeros((p * n, rank), dtype=np.int64)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
