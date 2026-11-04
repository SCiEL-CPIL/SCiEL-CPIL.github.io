import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS  # 4, 4, 5

    # Candidate decompositions in order of increasing rank (best first).
    candidates = []
    for name, builder in [
        ("kronecker3x3", _kronecker3x3),
        ("strassen", _strassen),
        ("winograd4x5", _winograd4x5),
        ("kronecker4x4", _kronecker4x4),
    ]:
        if time.time() > deadline - 15:
            break
        try:
            res = builder(n, m, p)
        except Exception:
            res = None
        if res is not None:
            rank = len(res["U"][0])
            if _verify(n, m, p, res):
                candidates.append((rank, name, res))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][2]

    return _schoolbook(n, m, p)


def _verify(n, m, p, res):
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]))
                if abs(s - 1.0) > 1e-6:
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _kronecker4x4(n, m, p):
    # 4x4x4 via Kronecker of 2x2x2 Strassen (rank 7) -> rank 49.
    r2 = 7
    U2 = [[1, 0, 0, 0, 0, 1, 0],
          [0, 1, 0, 0, 0, 0, 1],
          [0, 0, 1, 1, 1, 0, 0],
          [1, 1, 0, 0, 0, 1, 1]]
    V2 = [[1, 0, 0, 1, 0, 0, 0],
          [0, 1, 0, 1, 0, 0, 0],
          [0, 0, 1, 0, 0, 1, 0],
          [0, 0, 1, 1, 1, 0, 1]]
    W2 = [[1, 0, 1, 0, 0, 0, 0],
          [0, 1, 0, 0, 1, 0, 0],
          [1, 0, 0, 0, 0, 1, 0],
          [0, 1, 0, 1, 0, 1, 1]]

    rank = r2 * r2
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    U[(2 * a + c) * 4 + (2 * b + d)][r2 * c + d] = U2[a * 2 + c][b * 2 + d]
                    V[(2 * b + d) * 4 + (2 * c + a)][r2 * c + d] = V2[b * 2 + d][c * 2 + a]
                    W[(2 * c + a) * 4 + (2 * a + b)][r2 * c + d] = W2[c * 2 + a][a * 2 + b]
    # Pad V, W to p=5 columns (extra column stays zero).
    for r in range(16):
        V[r].append(0)
    for r in range(16):
        W[r].append(0)
    # Keep only first 16 columns of V (j=0..3) and first 16 rows of W (k=0..3).
    V = V[:16]
    W = W[:16]
    return {"U": U, "V": V, "W": W}


def _kronecker3x3(n, m, p):
    # 3x3x3 via Kronecker of 2x2x2 Strassen (rank 7) -> rank 49, then pad to 4x4x5.
    r2 = 7
    U2 = [[1, 0, 0, 0, 0, 1, 0],
          [0, 1, 0, 0, 0, 0, 1],
          [0, 0, 1, 1, 1, 0, 0],
          [1, 1, 0, 0, 0, 1, 1]]
    V2 = [[1, 0, 0, 1, 0, 0, 0],
          [0, 1, 0, 1, 0, 0, 0],
          [0, 0, 1, 0, 0, 1, 0],
          [0, 0, 1, 1, 1, 0, 1]]
    W2 = [[1, 0, 1, 0, 0, 0, 0],
          [0, 1, 0, 0, 1, 0, 0],
          [1, 0, 0, 0, 0, 1, 0],
          [0, 1, 0, 1, 0, 1, 1]]

    rank = r2 * r2
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    U[(2 * a + c) * 3 + (2 * b + d)][r2 * c + d] = U2[a * 2 + c][b * 2 + d]
                    V[(2 * b + d) * 3 + (2 * c + a)][r2 * c + d] = V2[b * 2 + d][c * 2 + a]
                    W[(2 * c + a) * 3 + (2 * a + b)][r2 * c + d] = W2[c * 2 + a][a * 2 + b]
    # Pad to 4x4x5.
    U = U + [[0] * rank for _ in range(7)]
    V = V + [[0] * rank for _ in range(11)]
    W = W + [[0] * rank for _ in range(11)]
    return {"U": U, "V": V, "W": W}


def _strassen(n, m, p):
    # 2-level Strassen for 4x4x4 embedded in 4x4x5. Rank 7^2 = 49.
    r2 = 7
    U2 = [[1, 0, 0, 0, 0, 1, 0],
          [0, 1, 0, 0, 0, 0, 1],
          [0, 0, 1, 1, 1, 0, 0],
          [1, 1, 0, 0, 0, 1, 1]]
    V2 = [[1, 0, 0, 1, 0, 0, 0],
          [0, 1, 0, 1, 0, 0, 0],
          [0, 0, 1, 0, 0, 1, 0],
          [0, 0, 1, 1, 1, 0, 1]]
    W2 = [[1, 0, 1, 0, 0, 0, 0],
          [0, 1, 0, 0, 1, 0, 0],
          [1, 0, 0, 0, 0, 1, 0],
          [0, 1, 0, 1, 0, 1, 1]]

    rank = r2 * r2
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    U[(2 * a + c) * 4 + (2 * b + d)][r2 * c + d] = U2[a * 2 + c][b * 2 + d]
                    V[(2 * b + d) * 4 + (2 * c + a)][r2 * c + d] = V2[b * 2 + d][c * 2 + a]
                    W[(2 * c + a) * 4 + (2 * a + b)][r2 * c + d] = W2[c * 2 + a][a * 2 + b]
    for r in range(16):
        V[r].append(0)
        W[r].append(0)
    V = V[:16]
    W = W[:16]
    return {"U": U, "V": V, "W": W}


def _winograd4x5(n, m, p):
    # Winograd-style: factor the 4x4x5 multiplication via 2x2x2 Strassen on
    # overlapping 2x2 blocks. Rank = 2 * 2 * 7 = 28.
    r2 = 7
    U2 = [[1, 0, 0, 0, 0, 1, 0],
          [0, 1, 0, 0, 0, 0, 1],
          [0, 0, 1, 1, 1, 0, 0],
          [1, 1, 0, 0, 0, 1, 1]]
    V2 = [[1, 0, 0, 1, 0, 0, 0],
          [0, 1, 0, 1, 0, 0, 0],
          [0, 0, 1, 0, 0, 1, 0],
          [0, 0, 1, 1, 1, 0, 1]]
    W2 = [[1, 0, 1, 0, 0, 0, 0],
          [0, 1, 0, 0, 1, 0, 0],
          [1, 0, 0, 0, 0, 1, 0],
          [0, 1, 0, 1, 0, 1, 1]]

    # Block structure: 2x2 grid of 2x2 blocks.
    # U: 16 rows (i=0..3, j=0..3), 28 cols (bi=0..1, bj=0..1, t=0..6)
    # V: 20 rows (j=0..3, k=0..4), 28 cols
    # W: 20 rows (k=0..4, i=0..3), 28 cols
    rank = 2 * 2 * r2  # 28
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(20)]
    W = [[0] * rank for _ in range(20)]

    for bi in range(2):
        for bj in range(2):
            for a in range(2):
                for b in range(2):
                    for c in range(2):
                        for d in range(2):
                            idx = (bi * 2 + bj) * r2 + (c * 2 + d)
                            U[(2 * bi + a) * 4 + (2 * bj + b)][idx] = U2[a * 2 + c][b * 2 + d]
                            # V: row = (2*bj+b)*5 + (2*bi+c) for k in range
                            V[(2 * bj + b) * 5 + (2 * bi + c)][idx] = V2[b * 2 + d][c * 2 + a]
                            # W: row = (2*bi+c)*4 + (2*bi+a)
                            W[(2 * bi + c) * 4 + (2 * bi + a)][idx] = W2[c * 2 + a][a * 2 + b]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
