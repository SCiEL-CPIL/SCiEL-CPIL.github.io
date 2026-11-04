import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_222():
    # U rows i*2+j, V rows j*2+k, W rows k*2+i  (2x2x2, rank 7)
    U = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, -1, 0],
    ], dtype=float).T  # 4x7
    V = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ], dtype=float).T  # 4x7
    W = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ], dtype=float).T  # 4x7
    return U, V, W


def _rect_224():
    # 2x2x4 = two independent Strassen blocks (left/right 2x2 of B). rank 14.
    Us, Vs, Ws = _strassen_222()
    Ur = np.hstack([Us, Us])                      # 4x14
    Vr = np.zeros((8, 14))                        # rows j*4+k
    for j in range(2):
        for k in range(2):
            Vr[j * 4 + k, 0:7] = Vs[j * 2 + k, :]
            Vr[j * 4 + (k + 2), 7:14] = Vs[j * 2 + k, :]
    Wr = np.zeros((8, 14))                        # rows k*2+i
    for k in range(2):
        for i in range(2):
            Wr[k * 2 + i, 0:7] = Ws[k * 2 + i, :]
            Wr[k * 2 + i, 7:14] = Ws[k * 2 + i, :]
    return Ur, Vr, Wr


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p):
    rng = np.random.default_rng(12345)
    for _ in range(3):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_true = A @ B
        C = np.zeros((n, p))
        for c in range(U.shape[1]):
            a = U[:, c].reshape(n, m) @ A
            b = V[:, c].reshape(m, p) @ B
            C += a * b * W[:, c].reshape(p, n).T
        if not np.allclose(C, C_true, atol=1e-8):
            return False
    return True


def construct():
    import time
    t0 = time.time()
    n, m, p = DIMENSIONS
    try:
        Us, Vs, Ws = _strassen_222()
        Ur, Vr, Wr = _rect_224()
        U = np.kron(Us, Ur)   # (16, 98)
        V = np.kron(Vs, Vr)   # (32, 98)
        W = np.kron(Ws, Wr)   # (32, 98)
        # Row permutations to the official index convention.
        permU = {}
        for i1 in range(2):
            for i2 in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        permU[i1 * 8 + i2 * 4 + j1 * 2 + j2] = i1 * 8 + j1 * 4 + i2 * 2 + j2
        U = U[[permU[y] for y in range(16)]]
        permV = {}
        for j1 in range(2):
            for j2 in range(2):
                for k1 in range(2):
                    for k2 in range(2):
                        permV[j1 * 16 + j2 * 8 + k1 * 2 + k2] = j1 * 16 + k1 * 8 + j2 * 4 + k2
        V = V[[permV[y] for y in range(32)]]
        permW = {}
        for k1 in range(2):
            for k2 in range(2):
                for i1 in range(2):
                    for i2 in range(2):
                        permW[k1 * 8 + k2 * 4 + i1 * 2 + i2] = k1 * 16 + i1 * 8 + k2 * 4 + i2
        W = W[[permW[y] for y in range(32)]]
        if time.time() - t0 < 110 and _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
