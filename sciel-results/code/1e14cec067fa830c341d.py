<<<<<<< SEARCH
    # Build rank-44 candidate:
    # - 4 blocks of columns for the 2x2 sub-block A01 x B0..1 (each block contributes 7 ranks) -> 28
    # - The remaining third A column (a=2) with B row a=2: schoolbook 2x8 -> 16
    rank = 28 + 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    # Part 1: A[0:2, 0:2] * B[0:2, :8]
    for block_idx in range(4):
        k0 = 2 * block_idx
        for r in range(7):
            # U: local (li,la) -> global row = li*m + la
            for li in range(2):
                for la in range(2):
                    uval = S_U[r][li * 2 + la]
                    if uval != 0:
                        U[li * m + la][col] = uval
            # V: local (la,lk) -> global row = la*p + (k0 + lk)
            for la in range(2):
                for lk in range(2):
                    vval = S_V[r][la * 2 + lk]
                    if vval != 0:
                        V[la * p + (k0 + lk)][col] = vval
            # W: local (lk,li) -> global row = (k0 + lk)*n + li
            for lk in range(2):
                for li in range(2):
                    wval = S_W[r][lk * 2 + li]
                    if wval != 0:
                        W[(k0 + lk) * n + li][col] = wval
            col += 1
            if not _ok():
                return _schoolbook()

    # Part 2: A[0:2, 2] * B[2, :8] (schoolbook, 16 terms)
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            if not _ok():
                return _schoolbook()

    # Deterministic exact check using integers only (values are integers)
    def _validate_int(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        R = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    s = 0
                    ur = U[i * m + a]
                    vr = V[a * p + k]
                    wr = W[k * n + i]
                    for c in range(R):
                        uv = ur[c]
                        if uv == 0:
                            continue
                        vv = vr[c]
                        if vv == 0:
                            continue
                        s += uv * vv * wr[c]
                    if s != 1:
                        return False
        return True

    cand = {"U": U, "V": V, "W": W}
    if _validate_int(cand):
        return cand
    # If something unexpected, fall back to safe schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
=======
    # Fallback: rank-44 Strassen+schoolbook
    rank = 28 + 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for block_idx in range(4):
        k0 = 2 * block_idx
        for r in range(7):
            for li in range(2):
                for la in range(2):
                    uval = S_U[r][li * 2 + la]
                    if uval != 0:
                        U[li * m + la][col] = uval
            for la in range(2):
                for lk in range(2):
                    vval = S_V[r][la * 2 + lk]
                    if vval != 0:
                        V[la * p + (k0 + lk)][col] = vval
            for lk in range(2):
                for li in range(2):
                    wval = S_W[r][lk * 2 + li]
                    if wval != 0:
                        W[(k0 + lk) * n + li][col] = wval
            col += 1
            if not _ok():
                return _schoolbook()

    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            if not _ok():
                return _schoolbook()

    def _validate_int(candidate):
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        R = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    s = 0
                    ur = U[i * m + a]
                    vr = V[a * p + k]
                    wr = W[k * n + i]
                    for c in range(R):
                        uv = ur[c]
                        if uv == 0:
                            continue
                        vv = vr[c]
                        if vv == 0:
                            continue
                        s += uv * vv * wr[c]
                    if s != 1:
                        return False
        return True

    cand = {"U": U, "V": V, "W": W}
    if _validate_int(cand):
        return cand
    return _schoolbook()
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
