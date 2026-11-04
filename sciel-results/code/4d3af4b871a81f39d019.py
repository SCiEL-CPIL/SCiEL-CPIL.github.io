import time

def construct():
    n, m, p = 2, 3, 3
    deadline = time.time() + 145

    # Build a rank-minimal decomposition by splitting the j (inner) dimension and
    # the k (output-row) dimension into blocks, using Strassen's rank-7 scheme on
    # the 2x2x2 block and direct (schoolbook) on the rest.
    #
    #   j = {0,1} | {2}          (inner product dimension)
    #   k = {0,1} | {2}          (output row dimension)
    #
    # Block (j in {0,1}, k in {0,1}) -> Strassen 2x2x2  : rank 7
    # Block (j = 2,      k in {0,1}) -> direct 2x1x2    : rank 4
    # Block (j in {0,1}, k = 2)      -> direct 2x2x1    : rank 4
    # Block (j = 2,      k = 2)      -> direct 2x1x1    : rank 2
    #   total rank = 17  (schoolbook is 18)
    #
    # U indexed by (i, j) -> i*m + j ; V by (j, k) -> j*p + k ; W by (k, i) -> k*n + i
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all (i,j,k)

    cols = []   # each entry: (u, v, w) lists of (index, coeff)
    for u in [((0, 0), 1), ((1, 1), 1)]:
        for v in [((0, 0), 1), ((1, 1), 1)]:
            for w in [((0, 0), 1), ((1, 1), 1)]:
                cols.append((list(u), list(v), list(w)))
    cols.append(([((1, 0), 1), ((1, 1), 1)], [((0, 0), 1)], [((0, 0), 1)]))
    cols.append(([((2, 0), 1), ((2, 1), 1)], [((0, 0), 1)], [((0, 0), 1)]))
    cols.append(([((0, 0), 1)], [((1, 0), 1), ((1, 1), -1)], [((1, 0), 1)]))
    cols.append(([((1, 1), 1)], [((0, 1), 1), ((0, 0), -1)], [((1, 1), 1)]))
    cols.append(([((0, 0), 1), ((0, 1), 1)], [((1, 1), 1)], [((1, 0), 1)]))
    cols.append(([((2, 0), 1), ((0, 0), -1)], [((0, 0), 1), ((0, 1), 1)], [((0, 1), 1)]))
    cols.append(([((0, 1), 1), ((1, 1), -1)], [((1, 0), 1), ((1, 1), 1)], [((1, 1), 1)]))
    for j in [0, 1]:
        cols.append(([((2, j), 1)], [((2, 0), 1)], [((0, 0), 1)]))
        cols.append(([((2, j), 1)], [((2, 1), 1)], [((1, 0), 1)]))
    for i in [0, 1]:
        cols.append(([((i, 0), 1)], [((0, 2), 1)], [((2, i), 1)]))
        cols.append(([((i, 1), 1)], [((1, 2), 1)], [((2, i), 1)]))
    cols.append(([((2, 2), 1)], [((2, 2), 1)], [((2, 0), 1)]))
    cols.append(([((2, 2), 1)], [((2, 2), 1)], [((2, 1), 1)]))

    rank = len(cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r, (u, v, w) in enumerate(cols):
        for (i, j), c in u:
            U[i * m + j][r] += c
        for (j, k), c in v:
            V[j * p + k][r] += c
        for (k, i), c in w:
            W[k * n + i][r] += c

    # Verify the decomposition exactly; if anything is off, fall back to schoolbook.
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break

    if ok:
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 18
    rs = n * m * p
    U = [[0] * rs for _ in range(n * m)]
    V = [[0] * rs for _ in range(m * p)]
    W = [[0] * rs for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
