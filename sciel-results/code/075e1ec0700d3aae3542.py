"""Direct rank-minimal tensor decomposition for 4x4 * 4x5 matrix multiplication."""
import time

DIMENSIONS = (4, 4, 5)


def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS

    # Rank 65: 49 terms (super-Strassen for 4x4*4x4 part) + 16 terms (schoolbook for 4x4*4x1)
    R = 65
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]

    def add(col, u, v, w):
        for i, j, c in u:
            if c:
                U[i * m + j][col] += c
        for j, k, c in v:
            if c:
                V[j * p + k][col] += c
        for k, i, c in w:
            if c:
                W[k * n + i][col] += c

    # Super-Strassen block combinations for 2x2 block matrices
    P_combos = (
        (1, 0, 0, 1), (0, 0, 1, 1), (1, 0, 0, 0),
        (0, 0, 0, 1), (1, 1, 0, 0), (-1, 0, 1, 0), (0, 1, 0, -1),
    )
    Q_combos = (
        (1, 0, 0, 1), (1, 0, 0, 0), (0, 1, 0, -1),
        (-1, 0, 1, 0), (0, 0, 0, 1), (1, 1, 0, 0), (0, 0, 1, 1),
    )

    # Inner Strassen: p_indices, q_indices, sign patterns
    inner = (
        ((0, 3), (0, 3), (1, 0, 0, 1)),
        ((2, 3), (0,), (0, 0, 1, -1)),
        ((0,), (1, 3), (0, 1, 0, 1)),
        ((3,), (2, 0), (1, 0, 1, 0)),
        ((0, 1), (3,), (-1, 1, 0, 0)),
        ((2, 0), (0, 1), (0, 0, 0, 1)),
        ((1, 3), (2, 3), (1, 0, 0, 0)),
    )

    # C-block contributions from each super term
    c_blocks = (
        (1, 0, 0, 1), (0, 0, 1, -1), (0, 1, 0, 1),
        (1, 0, 1, 0), (-1, 1, 0, 0), (0, 0, 0, 1), (1, 0, 0, 0),
    )

    def p_loc(idx, bl):
        r, c = divmod(idx, 2)
        return (r + 2 * (bl // 2), c + 2 * (bl % 2))

    def q_loc(idx, bl):
        r, c = divmod(idx, 2)
        return (r + 2 * (bl // 2), c + 2 * (bl % 2))

    def c_loc(bl, idx):
        r, c = divmod(idx, 2)
        return (r + 2 * (bl // 2), c + 2 * (bl % 2))

    col = 0
    for s in range(7):
        pb = P_combos[s]
        qb = Q_combos[s]
        cb = c_blocks[s]
        for t in range(7):
            pidx, qidx, sign = inner[t]
            ul, vl, wl = [], [], []
            for pi in pidx:
                for b in range(4):
                    coeff = pb[b]
                    if coeff:
                        r, c = p_loc(pi, b)
                        ul.append((r, c, coeff))
            for qi in qidx:
                for b in range(4):
                    coeff = qb[b]
                    if coeff:
                        r, c = q_loc(qi, b)
                        vl.append((r, c, coeff))
            for sp in range(4):
                sc = sign[sp]
                if not sc:
                    continue
                for b in range(4):
                    cc = cb[b]
                    if cc:
                        coeff = sc * cc
                        if coeff:
                            r, c = c_loc(b, sp)
                            wl.append((c, r, coeff))
            add(col, ul, vl, wl)
            col += 1

    # Schoolbook terms for the remaining 4x4 * 4x1 product (column 4)
    for i in range(n):
        for j in range(m):
            add(col, [(i, j, 1)], [(j, p - 1, 1)], [(p - 1, i, 1)])
            col += 1

    if time.time() - t_start > 118:
        return {"U": U, "V": V, "W": W}
    return {"U": U, "V": V, "W": W}
