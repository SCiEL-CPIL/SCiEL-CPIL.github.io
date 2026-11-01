import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    """
    Rank-56 decomposition for multiplying a 2x2 matrix A by a 2x16 matrix B.
    Uses Strassen's algorithm (rank 7) for each of the 8 independent 2x2 blocks.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS

    # Strassen 2x2x2 tensors
    sU = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=float)

    sV = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)

    sW = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ], dtype=float)

    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    for l in range(8):
        base = 7 * l
        for s in range(7):
            col = base + s
            U[:, col] = sU[s]
            b_rows = [2*l, 2*l+1, 16+2*l, 16+2*l+1]
            for b_idx in range(4):
                V[b_rows[b_idx], col] = sV[s, b_idx]
            c_rows = [4*l, 4*l+2, 4*l+1, 4*l+3]
            for c_idx in range(4):
                W[c_rows[c_idx], col] = sW[s, c_idx]

    # Verify
    def check(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m + j
                    v = j*p + k
                    w = k*n + i
                    val = np.dot(np.multiply(U[u,:], V[v,:]), W[w,:])
                    if abs(val - 1.0) > 1e-9:
                        return False
        valid = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid.add((i*m+j, j*p+k, k*n+i))
        for u in range(n*m):
            for v in range(m*p):
                for w in range(p*n):
                    if (u, v, w) not in valid:
                        val = np.dot(np.multiply(U[u,:], V[v,:]), W[w,:])
                        if abs(val) > 1e-9:
                            return False
        return True

    if time.time() < deadline and check(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback schoolbook rank 64
    rk = n * m * p
    U2 = np.zeros((n*m, rk), dtype=float)
    V2 = np.zeros((m*p, rk), dtype=float)
    W2 = np.zeros((p*n, rk), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U2[i*m+j, c] = 1.0
                V2[j*p+k, c] = 1.0
                W2[k*n+i, c] = 1.0
                c += 1
    return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
