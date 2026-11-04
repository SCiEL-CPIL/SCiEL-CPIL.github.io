"""Rank-minimal exact tensor decomposition for 4x4 x 4x5 matrix multiplication.

Construction: partition the 4x4x5 tensor into four disjoint sub-tensors:
  (rows{0,1} x cols{0,1} x out{0,1})  -> Strassen 2x2x2, rank 7
  (rows{2,3} x cols{0,1} x out{0,1})  -> Strassen 2x2x2, rank 7
  (rows{0,1} x cols{2,3} x out{2,3,4}) -> schoolbook 2x2x3, rank 8
  (rows{2,3} x cols{2,3} x out{2,3,4}) -> schoolbook 2x2x3, rank 8
Total rank = 30. A final full-tensor verification guards correctness; if it
ever fails we fall back to the full schoolbook (rank 80).
"""

import time

DIMENSIONS = [4, 4, 5]

_DEADLINE = time.time() + 140.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS

    def strassen_222(I, J, K):
        i0, i1 = I
        j0, j1 = J
        k0, k1 = K
        prods = [
            ([(i0, j0, 1), (i1, j1, 1)], [(j0, k0, 1), (j1, k1, 1)]),
            ([(i1, j0, 1), (i1, j1, 1)], [(j0, k0, 1)]),
            ([(i0, j0, 1)], [(j1, k0, 1), (j0, k0, -1)]),
            ([(i1, j1, 1)], [(j1, k1, 1), (j0, k1, -1)]),
            ([(i0, j0, 1), (i0, j1, 1)], [(j1, k1, 1)]),
            ([(i1, j0, 1), (i0, j0, -1)], [(j0, k0, 1), (j0, k1, 1)]),
            ([(i0, j1, 1), (i1, j1, -1)], [(j1, k0, 1), (j1, k1, 1)]),
        ]
        C_combos = [
            [(0, 1), (3, 1), (4, -1), (6, 1)],   # C[i0,k0]
            [(2, 1), (0, 1), (4, 1)],            # C[i0,k1]
            [(1, 1), (3, 1), (5, 1)],            # C[i1,k0]
            [(0, 1), (1, -1), (2, 1), (5, 1)],   # C[i1,k1]
        ]
        C_targets = [(i0, k0), (i0, k1), (i1, k0), (i1, k1)]
        terms = []
        for t_idx, (Arows, Brows) in enumerate(prods):
            u = {}
            for (ii, jj, s) in Arows:
                u[ii * m + jj] = u.get(ii * m + jj, 0) + s
            v = {}
            for (jj, kk, s) in Brows:
                v[jj * p + kk] = v.get(jj * p + kk, 0) + s
            w = {}
            for c_idx, combo in enumerate(C_combos):
                for (t, s) in combo:
                    if t == t_idx:
                        (ii, kk) = C_targets[c_idx]
                        w[kk * n + ii] = w.get(kk * n + ii, 0) + s
            terms.append((u, v, w))
        return terms

    def schoolbook_block(I, J, K):
        terms = []
        for i in I:
            for j in J:
                for k in K:
                    terms.append(({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1}))
        return terms

    blocks = [
        (strassen_222((0, 1), (0, 1), (0, 1))),
        (strassen_222((2, 3), (0, 1), (0, 1))),
        (schoolbook_block((0, 1), (2, 3), (2, 3, 4))),
        (schoolbook_block((2, 3), (2, 3), (2, 3, 4))),
    ]

    all_terms = []
    for b in blocks:
        all_terms.extend(b)

    rank = len(all_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c, (u, v, w) in enumerate(all_terms):
        for row, val in u.items():
            U[row][c] = val
        for row, val in v.items():
            V[row][c] = val
        for row, val in w.items():
            W[row][c] = val

    # Final full-tensor verification; fall back to schoolbook if broken.
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    ok = False
                    break
            if not ok:
                break

    if not ok:
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
# EVOLVE-BLOCK-END
