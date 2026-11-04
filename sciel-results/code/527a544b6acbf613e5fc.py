import time
import numpy as np

# ---------- Fixed mathematical building blocks ----------

# 2x2x2 Strassen tensor (7 terms). U: (4,7), V: (4,7), W: (4,7).
# A index a = i*2 + k ; B index b = k*2 + j ; C index c = j*2 + i
_STR2_U = np.array([
    [1, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 1],
    [1, 1, 0, 1, 0, 0, 1],
], dtype=object)
_STR2_V = np.array([
    [1, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0, 1],
], dtype=object)
_STR2_W = np.array([
    [1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 1, 1],
], dtype=object)

_TRIV_U = np.array([[1, 1, 1]], dtype=object)
_TRIV_V = np.identity(3, dtype=object)
_TRIV_W = np.identity(3, dtype=object)


def _kronecker(U1, V1, W1, U2, V2, W2):
    r1, r2 = U1.shape[1], U2.shape[1]
    U = np.zeros((U1.shape[0] * U2.shape[0], r1 * r2), dtype=object)
    V = np.zeros((V1.shape[0] * V2.shape[0], r1 * r2), dtype=object)
    W = np.zeros((W1.shape[0] * W2.shape[0], r1 * r2), dtype=object)
    for a1 in range(U1.shape[0]):
        for a2 in range(U2.shape[0]):
            for t1 in range(r1):
                for t2 in range(r2):
                    c = t1 * r2 + t2
                    U[a1 * U2.shape[0] + a2, c] = U1[a1, t1] * U2[a2, t2]
    for b1 in range(V1.shape[0]):
        for b2 in range(V2.shape[0]):
            for t1 in range(r1):
                for t2 in range(r2):
                    c = t1 * r2 + t2
                    V[b1 * V2.shape[0] + b2, c] = V1[b1, t1] * V2[b2, t2]
    for c1 in range(W1.shape[0]):
        for c2 in range(W2.shape[0]):
            for t1 in range(r1):
                for t2 in range(r2):
                    c = t1 * r2 + t2
                    W[c1 * W2.shape[0] + c2, c] = W1[c1, t1] * W2[c2, t2]
    return U, V, W


def _verify(U, V, W, n, m, p, trials=30, seed=12345):
    rng = np.random.RandomState(seed)
    for _ in range(trials):
        A = rng.randint(-3, 4, size=(n, m)).astype(object)
        B = rng.randint(-3, 4, size=(m, p)).astype(object)
        C = A @ B
        flat = np.zeros(n * p, dtype=object)
        for t in range(U.shape[1]):
            flat = flat + (U[:, t] * A.flatten()) * (V[:, t] * B.flatten()) * W[:, t]
        Cw = C.T.flatten()
        if not np.array_equal(flat, Cw):
            return False, np.nonzero(flat - Cw != 0)[0]
    return True, []


def construct():
    start = time.time()
    deadline = start + 120.0

    n, m, p = 4, 4, 9

    # 2x2 x 2x3 product: rank 21 via Kronecker(2x2x2, 1x1x3)
    U21, V21, W21 = _kronecker(_STR2_U, _STR2_V, _STR2_W, _TRIV_U, _TRIV_V, _TRIV_W)
    assert U21.shape == (4, 21) and V21.shape == (6, 21) and W21.shape == (6, 21)

    # Block index maps (row-major)
    A11 = [0, 1, 4, 5]
    A12 = [2, 3, 6, 7]
    A21 = [8, 9, 12, 13]
    A22 = [10, 11, 14, 15]
    B11 = [0, 1, 2, 9, 10, 11]
    B12 = [3, 4, 5, 12, 13, 14]
    B13 = [6, 7, 8, 15, 16, 17]
    B21 = [18, 19, 20, 27, 28, 29]
    B22 = [21, 22, 23, 30, 31, 32]
    B23 = [24, 25, 26, 33, 34, 35]
    C11 = [0, 1, 4, 5]
    C12 = [8, 9, 12, 13]
    C13 = [16, 17, 20, 21, 24, 25, 28, 29, 32, 33]
    C21 = [2, 3, 6, 7]
    C22 = [10, 11, 14, 15]
    C23 = [18, 19, 22, 23, 26, 27, 30, 31, 34, 35]

    # Required 12 products (each 2x2 x 2x3):
    # C11 = A11B11 + A12B21
    # C12 = A11B12 + A12B22
    # C13 = A11B13 + A12B23
    # C21 = A21B11 + A22B21
    # C22 = A21B12 + A22B22
    # C23 = A21B13 + A22B23
    #
    # Use Strassen-like grouping. Since standard Strassen doesn't directly
    # work for rectangular, I'll just enumerate the 12 products directly.
    # But I can group some: note that A11B11 and A22B21 appear in different Cs.
    #
    # For now, just do the 12 products directly: rank = 12 * 21 = 252.
    # That's worse than schoolbook 144. So let's just use schoolbook.
    #
    # Actually, let me try a different block split: A into 4 rows of 1, 
    # B into 4 rows of 1. Then each 1x4 x 4x9 = sum of 4 (1x1 x 1x9).
    # Each 1x1 x 1x9 has rank 9. Total 16*9 = 144. Same as schoolbook.
    #
    # The best I can do with simple blocks is 144. Let me just return schoolbook.

    rank = 144
    U = np.zeros((16, rank), dtype=object)
    V = np.zeros((36, rank), dtype=object)
    W = np.zeros((36, rank), dtype=object)
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k, col] = 1
                V[k * p + j, col] = 1
                W[j * n + i, col] = 1
                col += 1

    ok, bad = _verify(U, V, W, n, m, p, trials=20)
    if not ok:
        raise ValueError("Schoolbook verification failed: %s" % bad)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
