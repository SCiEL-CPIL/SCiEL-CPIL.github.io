import time

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 130.0

    def _elapsed_ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    # Schoolbook fallback
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _validate(candidate):
        from fractions import Fraction
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    u_row = U[i * m + a]
                    v_row = V[a * p + k]
                    w_row = W[k * n + i]
                    for c in range(rank):
                        if u_row[c] == 0 or v_row[c] == 0 or w_row[c] == 0:
                            continue
                        acc += Fraction(u_row[c]) * Fraction(v_row[c]) * Fraction(w_row[c])
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 decomposition
    # U indices: (li, la) -> li*2+la
    # V indices: (la, lk) -> la*2+lk
    # W indices: (lk, li) -> lk*2+li
    S_U = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    # Precompute non-zero entries for Strassen to speed up construction
    # For each rank r, store list of (u_idx, val), (v_idx, val), (w_idx, val)
    strassen_entries = []
    for r in range(7):
        u_entries = []
        for li in range(2):
            for la in range(2):
                val = S_U[r][li * 2 + la]
                if val != 0:
                    u_entries.append((li * m + la, val))
        v_entries = []
        for la in range(2):
            for lk in range(2):
                val = S_V[r][la * 2 + lk]
                if val != 0:
                    v_entries.append((la * p + lk, val))
        w_entries = []
        for lk in range(2):
            for li in range(2):
                val = S_W[r][lk * 2 + li]
                if val != 0:
                    w_entries.append((lk * n + li, val))
        strassen_entries.append((u_entries, v_entries, w_entries))

    # Verify Strassen decomposition
    def _verify_strassen():
        for i in range(2):
            for a in range(2):
                for k in range(2):
                    acc = 0
                    for r in range(7):
                        acc += S_U[r][i * 2 + a] * S_V[r][a * 2 + k] * S_W[r][k * 2 + i]
                    if acc != 1:
                        return False
        return True

    if not _verify_strassen():
        return _schoolbook()

    # Build candidate (Rank 44)
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    # Part 1: 4 blocks of Strassen for rows 0,1 of A
    for block_idx in range(4):
        k0 = block_idx * 2
        for r in range(7):
            u_entries, v_entries, w_entries = strassen_entries[r]
            for (u_idx, val) in u_entries:
                U[u_idx][col] = val
            for (v_idx_local, val) in v_entries:
                # v_idx_local has k in 0,1; need to offset by k0
                a_local = v_idx_local // p
                k_local = v_idx_local % p
                v_idx = a_local * p + (k0 + k_local)
                V[v_idx][col] = val
            for (w_idx_local, val) in w_entries:
                # w_idx_local has k in 0,1; need to offset by k0
                k_local = w_idx_local // n
                i_local = w_idx_local % n
                w_idx = (k0 + k_local) * n + i_local
                W[w_idx][col] = val
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    # Part 2: Schoolbook for row 2 of A
    for i in range(2):
        for k in range(8):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            if not _elapsed_ok():
                return _schoolbook()

    candidate = {"U": U, "V": V, "W": W}

    if _validate(candidate):
        return candidate
    else:
        return _schoolbook()

# EVOLVE-BLOCK-END
