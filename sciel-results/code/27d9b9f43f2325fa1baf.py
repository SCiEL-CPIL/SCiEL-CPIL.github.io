import numpy as np
import time

_START = time.time()
_DEADLINE = 120.0

def construct():
    n, m, p = 4, 5, 5
    cols = []

    def add(u, v, w):
        cols.append((u, v, w))

    def aidx(i, k):
        return i * m + k

    def bidx(k, j):
        return k * p + j

    def cidx(j, i):
        return j * n + i

    def add_entry(u, v, w, i, k, j, s):
        u[aidx(i, k)] += s
        v[bidx(k, j)] += s
        w[cidx(j, i)] += s

    def add_sum(u, v, w, A, B, C, s):
        for (i, k), sv in A.items():
            u[aidx(i, k)] += s * sv
        for (k, j), sv in B.items():
            v[bidx(k, j)] += s * sv
        for (j, i), sv in C.items():
            w[cidx(j, i)] += s * sv

    # ---- Outer 4x4 Strassen over the (i,k) x (k,j) x (i,j) core (j,k in 0..3) ----
    # Each factor is a product of a block-level factor (4x4 Strassen) and an
    # inner 2x2 Strassen factor acting within each 2x2 block.
    outer = [
        ([(0, 1)], [(0, 1)], [(0, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    inner = [
        ((0, 1), (0, 1), (0, 1)),
        ((0, 1), (0, 1), (1, 1)),
        ((0, 1), (1, -1), (0, 1)),
        ((1, 1), (0, -1), (1, 1)),
        ((0, 1), (1, 1), (0, 1)),
        ((0, -1), (0, 1), (1, 1)),
        ((1, -1), (1, 1), (0, 1)),
    ]

    for (Ab, Bb, Cb) in outer:
        for (A2, B2, C2) in inner:
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            for (b, s) in Ab:
                for (li, lk) in A2:
                    add_entry(u, v, w, 2 * b + li, 2 * b + lk, 0, s)
            for (b, s) in Bb:
                for (lk, lj) in B2:
                # B factor index: k=2b+lk, j=2b+lj
                    v[bidx(2 * b + lk, 2 * b + lj)] += s
            for (b, s) in Cb:
                for (li, lj) in C2:
                    w[cidx(2 * b + lj, 2 * b + li)] += s
            add(u, v, w)

    # ---- Boundary j=4:  C[i,4] = A[i,4]B[4,4] + (A[i,3]+A[i,4])B[3,4] - A[i,3]B[3,4] ----
    for i in range(4):
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
        u[aidx(i, 4)] += 1
        v[bidx(4, 4)] += 1
        w[cidx(4, i)] += 1
        add(u, v, w)

        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
        u[aidx(i, 3)] += 1; u[aidx(i, 4)] += 1
        v[bidx(3, 4)] += 1
        w[cidx(4, i)] += 1
        add(u, v, w)

        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
        u[aidx(i, 3)] += 1
        v[bidx(3, 4)] += 1
        w[cidx(4, i)] += -1
        add(u, v, w)

    # ---- Boundary k=4:  C[i,j] = A[i,4]B[4,j] + (B[3,j]+B[4,j])A[i,4] - B[3,j]A[i,4] ----
    for j in range(5):
        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
        u[aidx(0, 4)] += 1
        v[bidx(4, j)] += 1
        w[cidx(j, 0)] += 1
        add(u, v, w)

        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
        u[aidx(0, 4)] += 1
        v[bidx(3, j)] += 1; v[bidx(4, j)] += 1
        w[cidx(j, 0)] += 1
        add(u, v, w)

        u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
        u[aidx(0, 4)] += 1
        v[bidx(3, j)] += 1
        w[cidx(j, 0)] += -1
        add(u, v, w)

    rank = len(cols)
    if rank == 0:
        add(np.ones(n * m), np.ones(m * p), np.ones(p * n))
        rank = 1

    U = np.stack([c[0] for c in cols]).T
    V = np.stack([c[1] for c in cols]).T
    W = np.stack([c[2] for c in cols]).T
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
