import time

DIMENSIONS = [3, 4, 5]

def construct():
    start = time.time()
    deadline = start + 115.0
    n, m, p = DIMENSIONS

    # Block partition to maximize Strassen 2x2x2 usage
    # i: [0,2), [2,3)  -> sizes 2, 1
    # j: [0,2), [2,4)  -> sizes 2, 2
    # k: [0,2), [2,4), [4,5) -> sizes 2, 2, 1

    i_blocks = [(0, 2), (2, 3)]
    j_blocks = [(0, 2), (2, 4)]
    k_blocks = [(0, 2), (2, 4), (4, 5)]

    u_entries = []
    v_entries = []
    w_entries = []
    col_idx = 0

    # Strassen 2x2x2 terms (local indices: u/v are [row*2+col], w is [k*2+i])
    strassen_terms = [
        # (u, v, w) for M1..M7
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1=(a+d)(e+h) -> C00,C11
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),  # M2=(c+d)e    -> C10
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),  # M3=a(f-h)    -> C01,C11
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),  # M4=d(g-e)    -> C00,C10
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5=(a+b)h    -> C00,C01
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6=(c-a)(e+f)-> C11
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1]),  # M7=(b-d)(g+h)-> C00,C11
    ]

    for i0, i1 in i_blocks:
        ni = i1 - i0
        for j0, j1 in j_blocks:
            nj = j1 - j0
            for k0, k1 in k_blocks:
                nk = k1 - k0

                if ni == 2 and nj == 2 and nk == 2:
                    for u_loc, v_loc, w_loc in strassen_terms:
                        for li in range(2):
                            for lj in range(2):
                                val = u_loc[li * 2 + lj]
                                if val != 0:
                                    u_entries.append(((i0 + li) * m + (j0 + lj), col_idx, val))
                        for lj in range(2):
                            for lk in range(2):
                                val = v_loc[lj * 2 + lk]
                                if val != 0:
                                    v_entries.append(((j0 + lj) * p + (k0 + lk), col_idx, val))
                        for lk in range(2):
                            for li in range(2):
                                val = w_loc[lk * 2 + li]
                                if val != 0:
                                    w_entries.append(((k0 + lk) * n + (i0 + li), col_idx, val))
                        col_idx += 1
                else:
                    for li in range(ni):
                        for lj in range(nj):
                            for lk in range(nk):
                                u_entries.append(((i0 + li) * m + (j0 + lj), col_idx, 1))
                                v_entries.append(((j0 + lj) * p + (k0 + lk), col_idx, 1))
                                w_entries.append(((k0 + lk) * n + (i0 + li), col_idx, 1))
                                col_idx += 1

    rank = col_idx

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r, c, val in u_entries:
        U[r][c] = val
    for r, c, val in v_entries:
        V[r][c] = val
    for r, c, val in w_entries:
        W[r][c] = val

    # Verify correctness
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True

    if not verify(U, V, W, rank):
        U, V, W = construct_fallback(n, m, p)

    return {"U": U, "V": V, "W": W}

def construct_fallback(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W
