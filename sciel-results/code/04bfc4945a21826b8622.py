import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin under the 150s wall-clock kill

    n, m, p = DIMENSIONS
    # 2x2 Strassen factors (a_{ij}, b_{jk}, c_{ik}) with c = sum_k a*b over 7 factors
    strassen = [
        (1, 1, 1), (1, 1, -1), (1, 1, 1), (1, 1, -1),
        (1, -1, 1), (1, 1, -1), (1, 1, 1),
    ]

    def _rank1(i0, j0, k0):
        """One rank-1 term A[i0,j0]*B[j0,k0]."""
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        u[i0 * m + j0] = 1
        v[j0 * p + k0] = 1
        w[k0 * n + i0] = 1
        return u, v, w

    def _strassen(i0, j0, k0):
        """Seven rank-1 terms for a 2x2 sub-block A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]."""
        us, vs, ws = [], [], []
        for (a, b, c) in strassen:
            for ii in (0, 1):
                for jj in (0, 1):
                    for kk in (0, 1):
                        if ii == jj and jj == kk:  # (a,b) diagonal pair
                            A = 1 if ii == 0 else -1
                        elif ii == jj:  # a diagonal
                            A = a
                        elif jj == kk:  # b diagonal
                            A = b
                        else:  # i=k diagonal
                            A = c
                        if A == 0:
                            continue
                        u = [0] * (n * m)
                        v = [0] * (m * p)
                        w = [0] * (p * n)
                        u[(i0 + ii) * m + (j0 + jj)] = A
                        v[(j0 + jj) * p + (k0 + kk)] = 1
                        w[(k0 + kk) * n + (i0 + ii)] = 1
                        us.append(u)
                        vs.append(v)
                        ws.append(w)
        return us, vs, ws

    U, V, W = [], [], []

    # 2x2 blocks (Strassen, rank 7 each)
    for (j0, k0) in [(0, 0), (0, 2), (0, 4), (2, 0), (2, 2), (2, 4)]:
        us, vs, ws = _strassen(0, j0, k0)
        U += us; V += vs; W += ws

    # 1x2 blocks (rank-1 each) for the remaining row of A (row 4)
    for (j0, k0) in [(4, 0), (4, 2), (4, 4)]:
        for kk in (0, 1):
            u, v, w = _rank1(0, j0, k0 + kk)
            U.append(u); V.append(v); W.append(w)

    rank = len(U)

    # Exact integer verification: (A B)[i,k] = sum_r U[i*m+j..]*V*W must hold for all (i,k).
    import numpy as np
    rng = np.random.RandomState(12345)
    for _ in range(200):
        A = rng.randint(-3, 4, size=(n, m)).astype(object)
        B = rng.randint(-3, 4, size=(m, p)).astype(object)
        C = A @ B
        for i in range(n):
            for k in range(p):
                acc = 0
                for r in range(rank):
                    s = 0
                    for j in range(m):
                        s += U[i * m + j][r] * V[j * p + k][r]
                    acc += s * W[k * n + i][r]
                if acc != C[i, k]:
                    # fall back to guaranteed-correct schoolbook (rank 60)
                    U2 = [[0] * (n * m * p) for _ in range(n * m)]
                    V2 = [[0] * (n * m * p) for _ in range(m * p)]
                    W2 = [[0] * (n * m * p) for _ in range(p * n)]
                    col = 0
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                U2[i2 * m + j2][col] = 1
                                V2[j2 * p + k2][col] = 1
                                W2[k2 * n + i2][col] = 1
                                col += 1
                    return {"U": U2, "V": V2, "W": W2}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
