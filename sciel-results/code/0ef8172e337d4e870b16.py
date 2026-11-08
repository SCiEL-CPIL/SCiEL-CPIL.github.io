import time

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    def _ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    nm, mp, pn = n * m, m * p, p * n

    def _schoolbook():
        R = nm * mp * pn
        U = [[0] * R for _ in range(nm)]
        V = [[0] * R for _ in range(mp)]
        W = [[0] * R for _ in range(pn)]
        c = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][c] = 1
                    V[a * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    if not _ok():
        return _schoolbook()

    # Strassen 2x2x2 core matrices
    # S_U[r][i*2+a], S_V[r][a*2+k], S_W[r][k*2+i]
    S_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    # Verify Strassen core is correct
    for i in range(2):
        for a in range(2):
            for k in range(2):
                s = 0
                for r in range(7):
                    s += S_U[r][i * 2 + a] * S_V[r][a * 2 + k] * S_W[r][k * 2 + i]
                if s != 1:
                    return _schoolbook()

    rank = 44
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    col = 0

    # Part 1: 4 Strassen blocks for A[0:2, 0:2] * B[0:2, :]
    for block_idx in range(4):
        if not _ok():
            return _schoolbook()
        k0 = block_idx * 2

        for r in range(7):
            c = col + r

            # U: rows i*m+a for i in 0,1 and a in 0,1
            # local index: i*2+a
            for li in range(2):
                for la in range(2):
                    val = S_U[r][li * 2 + la]
                    if val != 0:
                        U[li * m + la][c] = val

            # V: rows a*p+k for a in 0,1 and k in k0, k0+1
            # local index: a*2+k_rel
            for la in range(2):
                for lk in range(2):
                    val = S_V[r][la * 2 + lk]
                    if val != 0:
                        V[la * p + (k0 + lk)][c] = val

            # W: rows k*n+i for k in k0, k0+1 and i in 0,1
            # local index: k_rel*2+i
            for lk in range(2):
                for li in range(2):
                    val = S_W[r][lk * 2 + li]
                    if val != 0:
                        W[(k0 + lk) * n + li][c] = val

        col += 7

    if not _ok():
        return _schoolbook()

    # Part 2: Schoolbook for A[0:2, 2] * B[2, :8]
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

            if not _ok():
                return _schoolbook()

    # Exact integer validation
    for i in range(n):
        for a in range(m):
            for k in range(p):
                u_row = U[i * m + a]
                v_row = V[a * p + k]
                w_row = W[k * n + i]
                s = 0
                for c in range(rank):
                    u_val = u_row[c]
                    if u_val == 0:
                        continue
                    v_val = v_row[c]
                    if v_val == 0:
                        continue
                    w_val = w_row[c]
                    if w_val == 0:
                        continue
                    s += u_val * v_val * w_val
                if s != 1:
                    return _schoolbook()

    return {"U": U, "V": V, "W": W}
