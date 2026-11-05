import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    # Build rank-28 baseline
    rank_28 = 28
    U28 = np.zeros((n * m, rank_28))
    V28 = np.zeros((m * p, rank_28))
    W28 = np.zeros((p * n, rank_28))
    
    col = 0
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        U28[i1*m+j1, col] = 1; U28[i2*m+j2, col] = 1
        V28[j1*p+k1, col] = 1; V28[j2*p+k2, col] = 1
        W28[k1*n+i1, col] = 1; W28[k2*n+i2, col] = 1
        col += 1
        U28[i2*m+j1, col] = 1; U28[i2*m+j2, col] = 1
        V28[j1*p+k1, col] = 1
        W28[k1*n+i2, col] = 1; W28[k2*n+i2, col] = -1
        col += 1
        U28[i1*m+j1, col] = 1
        V28[j1*p+k2, col] = 1; V28[j2*p+k2, col] = -1
        W28[k2*n+i1, col] = 1; W28[k2*n+i2, col] = 1
        col += 1
        U28[i2*m+j2, col] = 1
        V28[j2*p+k1, col] = 1; V28[j1*p+k1, col] = -1
        W28[k1*n+i1, col] = 1; W28[k1*n+i2, col] = 1
        col += 1
        U28[i1*m+j1, col] = 1; U28[i1*m+j2, col] = 1
        V28[j2*p+k2, col] = 1
        W28[k1*n+i1, col] = -1; W28[k2*n+i1, col] = 1
        col += 1
        U28[i2*m+j1, col] = 1; U28[i1*m+j1, col] = -1
        V28[j1*p+k1, col] = 1; V28[j1*p+k2, col] = 1
        W28[k2*n+i2, col] = 1
        col += 1
        U28[i1*m+j2, col] = 1; U28[i2*m+j2, col] = -1
        V28[j2*p+k1, col] = 1; V28[j2*p+k2, col] = 1
        W28[k1*n+i1, col] = 1
        col += 1

    def add_schoolbook(i_indices, j_indices, k_indices):
        nonlocal col
        for i in i_indices:
            for j in j_indices:
                for k in k_indices:
                    U28[i*m+j, col] = 1
                    V28[j*p+k, col] = 1
                    W28[k*n+i, col] = 1
                    col += 1

    add_strassen(0, 1, 0, 1, 0, 1)
    add_strassen(0, 1, 0, 1, 2, 3)
    add_schoolbook([0, 1], [0, 1], [4])
    add_schoolbook([0, 1], [2], [0, 1])
    add_schoolbook([0, 1], [2], [2, 3])
    add_schoolbook([0, 1], [2], [4])

    def verify(U, V, W):
        res = np.zeros((n, m, p))
        for c in range(U.shape[1]):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            res += Uc[:, :, np.newaxis] * Vc[np.newaxis, :, :] * Wc.T[:, np.newaxis, :]
        target = np.ones((n, m, p))
        return np.allclose(res, target, atol=1e-9)

    if not verify(U28, V28, W28):
        U30 = np.zeros((n * m, 30))
        V30 = np.zeros((m * p, 30))
        W30 = np.zeros((p * n, 30))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U30[i*m+j, col] = 1
                    V30[j*p+k, col] = 1
                    W30[k*n+i, col] = 1
                    col += 1
        return {"U": U30.tolist(), "V": V30.tolist(), "W": W30.tolist()}

    # Try to reduce rank by removing one redundant term
    for remove_idx in range(28):
        if time.time() > deadline - 10:
            break
        U_test = np.delete(U28, remove_idx, axis=1)
        V_test = np.delete(V28, remove_idx, axis=1)
        W_test = np.delete(W28, remove_idx, axis=1)
        if verify(U_test, V_test, W_test):
            return {"U": U_test.tolist(), "V": V_test.tolist(), "W": W_test.tolist()}

    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
