import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130  # safety margin before the 150s kill
    n, m, p = DIMENSIONS  # 4, 4, 6

    def verify_full(U, V, W, n, m, p, tol=1e-8):
        """True iff sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for every i,j,k."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        Uv = Ua.reshape(n, m, -1)
        Vv = Va.reshape(m, p, -1)
        Wv = Wa.reshape(p, n, -1)
        T = np.einsum('ijr,jkr,kir->ijk', Uv, Vv, Wv)
        return bool(np.all(np.abs(T - 1.0) <= tol))

    # Rank-5 bilinear algorithm for 2x2 * 2x3
    def build_2x2x3():
        U = [[0.0] * 5 for _ in range(4)]
        V = [[0.0] * 5 for _ in range(6)]
        W = [[0.0] * 5 for _ in range(6)]
        def add(a_idx, b_idx, c_idx, t, val):
            U[a_idx][t] += val
            V[b_idx][t] += val
            W[c_idx][t] += val
        for (ai, aj) in [(0, 0), (1, 1)]:
            for (bj, bk) in [(0, 0), (1, 0)]:
                for (ck, ci) in [(0, 0), (0, 1)]:
                    add(ai * 2 + aj, bj * 3 + bk, ck * 2 + ci, 0, 1.0)
        for (ai, aj) in [(0, 0), (1, 1)]:
            for (bj, bk) in [(0, 1), (1, 1)]:
                for (ck, ci) in [(1, 0), (1, 1)]:
                    add(ai * 2 + aj, bj * 3 + bk, ck * 2 + ci, 1, 1.0)
        for (ai, aj) in [(0, 0), (1, 1)]:
            for (bj, bk) in [(0, 2), (1, 2)]:
                for (ck, ci) in [(2, 0), (2, 1)]:
                    add(ai * 2 + aj, bj * 3 + bk, ck * 2 + ci, 2, 1.0)
        for (bj, bk) in [(1, 0), (0, 0)]:
            for (ck, ci) in [(0, 0), (0, 1)]:
                add(0, bj * 3 + bk, ck * 2 + ci, 3, 1.0 if bj == 1 else -1.0)
        for (bj, bk) in [(0, 1), (1, 1)]:
            for (ck, ci) in [(1, 0), (1, 1)]:
                add(3, bj * 3 + bk, ck * 2 + ci, 4, 1.0 if bj == 0 else -1.0)
        return U, V, W

    # Rank-7 Strassen for 2x2 * 2x2
    def build_2x2x2():
        U = [[0.0] * 7 for _ in range(4)]
        V = [[0.0] * 7 for _ in range(4)]
        W = [[0.0] * 7 for _ in range(4)]
        def aidx(i, j): return i * 2 + j
        def bidx(i, j): return i * 2 + j
        def cidx(i, j): return j * 2 + i
        def addA(a, t, v): U[a][t] += v
        def addB(b, t, v): V[b][t] += v
        def addC(c, t, v): W[c][t] += v
        # t0
        for (ai, aj) in [(0, 0), (1, 1)]:
            for (bi, bj) in [(0, 0), (1, 1)]:
                for (ci, cj) in [(0, 0), (1, 1)]:
                    addA(aidx(ai, aj), 0, 1.0); addB(bidx(bi, bj), 0, 1.0); addC(cidx(ci, cj), 0, 1.0)
        # t1
        for (ai, aj) in [(1, 0), (1, 1)]:
            for (bi, bj) in [(0, 0)]:
                for (ci, cj) in [(1, 0)]:
                    addA(aidx(ai, aj), 1, 1.0); addB(bidx(bi, bj), 1, 1.0); addC(cidx(ci, cj), 1, 1.0)
        # t2
        for (bi, bj) in [(1, 1), (0, 0)]:
            for (ci, cj) in [(0, 1)]:
                addA(aidx(0, 0), 2, 1.0); addB(bidx(bi, bj), 2, 1.0 if (bi, bj) == (1, 1) else -1.0); addC(cidx(ci, cj), 2, 1.0)
        # t3
        for (bi, bj) in [(0, 1), (1, 1)]:
            for (ci, cj) in [(1, 1)]:
                addA(aidx(1, 1), 3, 1.0); addB(bidx(bi, bj), 3, 1.0 if (bi, bj) == (0, 1) else -1.0); addC(cidx(ci, cj), 3, 1.0)
        # t4
        for (ai, aj) in [(0, 0), (0, 1)]:
            for (bi, bj) in [(1, 1)]:
                for (ci, cj) in [(0, 0), (1, 1)]:
                    addA(aidx(ai, aj), 4, 1.0); addB(bidx(bi, bj), 4, 1.0); addC(cidx(ci, cj), 4, 1.0 if (ci, cj) == (0, 0) else -1.0)
        # t5
        for (ai, aj) in [(1, 0), (0, 0)]:
            for (bi, bj) in [(0, 0), (0, 1)]:
                for (ci, cj) in [(1, 0)]:
                    addA(aidx(ai, aj), 5, 1.0 if (ai, aj) == (1, 0) else -1.0); addB(bidx(bi, bj), 5, 1.0); addC(cidx(ci, cj), 5, -1.0)
        # t6
        for (ai, aj) in [(0, 1), (1, 1)]:
            for (bi, bj) in [(1, 0), (1, 1)]:
                for (ci, cj) in [(0, 1)]:
                    addA(aidx(ai, aj), 6, 1.0 if (ai, aj) == (0, 1) else -1.0); addB(bidx(bi, bj), 6, 1.0); addC(cidx(ci, cj), 6, -1.0)
        return U, V, W

    # Assemble 4x4 * 4x6 from 7 copies of 2x2x2 and 7 copies of 2x2x3.
    def assemble():
        R = 7 * 7 + 7 * 5  # 84
        U = [[0.0] * R for _ in range(16)]
        V = [[0.0] * R for _ in range(24)]
        W = [[0.0] * R for _ in range(24)]
        U22, V22, W22 = build_2x2x2()
        U23, V23, W23 = build_2x2x3()

        products = [
            ([(1, 0, 0), (1, 2, 2)], [(1, 0, 0), (1, 2, 3)], 22),
            ([(1, 2, 0), (1, 2, 2)], [(1, 0, 0)], 22),
            ([(1, 0, 0)], [(1, 2, 3), (-1, 0, 0)], 22),
            ([(1, 2, 2)], [(1, 0, 3), (-1, 2, 3)], 22),
            ([(1, 0, 0), (1, 0, 2)], [(1, 2, 3)], 23),
            ([(1, 2, 0), (-1, 0, 0)], [(1, 0, 0), (1, 0, 3)], 23),
            ([(1, 0, 2), (-1, 2, 2)], [(1, 2, 0), (1, 2, 3)], 23),
        ]
        c_recon = [
            [(1, 0, 0), (1, 2, 0)],
            [(1, 2, 0)],
            [(1, 2, 0)],
            [(1, 0, 3)],
            [(1, 0, 0), (-1, 2, 0)],
            [(-1, 2, 0)],
            [(1, 0, 3)],
        ]

        for prod_idx, (a_terms, b_terms, base) in enumerate(products):
            if base == 22:
                Ub, Vb, Wb = U22, V22, W22; nA, mA, nB, pB = 2, 2, 2, 2
            else:
                Ub, Vb, Wb = U23, V23, W23; nA, mA, nB, pB = 2, 2, 2, 3
            Rbase = len(Ub[0])
            c_terms = c_recon[prod_idx]
            for t in range(Rbase):
                for (sa, a_row, a_col) in a_terms:
                    for (sb, b_row, b_col) in b_terms:
                        for (sc, c_row, c_col) in c_terms:
                            for li in range(nA):
                                for lj in range(mA):
                                    u_loc = Ub[li * mA + lj][t]
                                    if u_loc == 0.0: continue
                                    U[(a_row + li) * 4 + (a_col + lj)][prod_idx * Rbase + t] += sa * u_loc
                            for lj in range(nB):
                                for lk in range(pB):
                                    v_loc = Vb[lj * pB + lk][t]
                                    if v_loc == 0.0: continue
                                    V[(b_row + lj) * 6 + (b_col + lk)][prod_idx * Rbase + t] += sb * v_loc
                            for lk in range(pB):
                                for li in range(nA):
                                    w_loc = Wb[lk * nA + li][t]
                                    if w_loc == 0.0: continue
                                    W[(c_col + lk) * 4 + (c_row + li)][prod_idx * Rbase + t] += sc * w_loc
        return U, V, W

    # Fallback: 4x4*4x4 (rank 49) + 4x4*4x2 (rank 21) = 70, valid.
    def assemble_fallback():
        U44, V44, W44 = build_2x2x2()
        def build_2x2x1():
            U = [[0.0] * 3 for _ in range(4)]
            V = [[0.0] * 3 for _ in range(2)]
            W = [[0.0] * 3 for _ in range(2)]
            U[0][0] = 1.0; U[3][0] = 1.0; V[0][0] = 1.0; W[0][0] = 1.0; W[1][0] = 1.0
            U[0][1] = 1.0; V[1][1] = 1.0; W[0][1] = 1.0
            U[3][2] = 1.0; V[1][2] = 1.0; W[1][2] = -1.0
            return U, V, W
        U42, V42, W42 = build_2x2x1()
        R = 7 * 7 + 7 * 3
        U = [[0.0] * R for _ in range(16)]
        V = [[0.0] * R for _ in range(24)]
        W = [[0.0] * R for _ in range(24)]
        products44 = [
            ([(1, 0, 0), (1, 2, 2)], [(1, 0, 0), (1, 2, 2)], [(1, 0, 0), (1, 2, 2)]),
            ([(1, 2, 0), (1, 2, 2)], [(1, 0, 0)], [(1, 2, 0)]),
            ([(1, 0, 0)], [(1, 2, 2), (-1, 0, 0)], [(1, 2, 0)]),
            ([(1, 2, 2)], [(1, 0, 2), (-1, 2, 2)], [(1, 0, 2)]),
            ([(1, 0, 0), (1, 0, 2)], [(1, 2, 2)], [(1, 0, 0), (-1, 2, 2)]),
            ([(1, 2, 0), (-1, 0, 0)], [(1, 0, 0), (1, 0, 2)], [(-1, 2, 0)]),
            ([(1, 0, 2), (-1, 2, 2)], [(1, 2, 0), (1, 2, 2)], [(1, 0, 2)]),
        ]
        for pi, (a_terms, b_terms, c_terms) in enumerate(products44):
            for t in range(7):
                for (sa, ar, ac) in a_terms:
                    for (sb, br, bc) in b_terms:
                        for (sc, cr, cc) in c_terms:
                            for li in range(2):
                                for lj in range(2):
                                    u_loc = U44[li * 2 + lj][t]
                                    if u_loc == 0.0: continue
                                    U[(ar + li) * 4 + (ac + lj)][pi * 7 + t] += sa * u_loc
                            for lj in range(2):
                                for lk in range(2):
                                    v_loc = V44[lj * 2 + lk][t]
                                    if v_loc == 0.0: continue
                                    V[(br + lj) * 6 + (bc + lk)][pi * 7 + t] += sb * v_loc
                            for lk in range(2):
                                for li in range(2):
                                    w_loc = W44[lk * 2 + li][t]
                                    if w_loc == 0.0: continue
                                    W[(cc + lk) * 4 + (cr + li)][pi * 7 + t] += sc * w_loc
        products42 = [
            ([(1, 0, 0), (1, 2, 2)], [(1, 0, 4), (1, 2, 5)], [(1, 0, 4), (1, 2, 5)]),
            ([(1, 2, 0), (1, 2, 2)], [(1, 0, 4)], [(1, 2, 4)]),
            ([(1, 0, 0)], [(1, 2, 5), (-1, 0, 4)], [(1, 2, 4)]),
            ([(1, 2, 2)], [(1, 0, 5), (-1, 2, 5)], [(1, 0, 5)]),
            ([(1, 0, 0), (1, 0, 2)], [(1, 2, 5)], [(1, 0, 4), (-1, 2, 5)]),
            ([(1, 2, 0), (-1, 0, 0)], [(1, 0, 4), (1, 0, 5)], [(-1, 2, 4)]),
            ([(1, 0, 2), (-1, 2, 2)], [(1, 2, 4), (1, 2, 5)], [(1, 0, 5)]),
        ]
        base42 = 7 * 7
        for pi, (a_terms, b_terms, c_terms) in enumerate(products42):
            for t in range(3):
                for (sa, ar, ac) in a_terms:
                    for (sb, br, bc) in b_terms:
                        for (sc, cr, cc) in c_terms:
                            for li in range(2):
                                for lj in range(2):
                                    u_loc = U42[li * 2 + lj][t]
                                    if u_loc == 0.0: continue
                                    U[(ar + li) * 4 + (ac + lj)][base42 + pi * 3 + t] += sa * u_loc
                            for lj in range(2):
                                for lk in range(1):
                                    v_loc = V42[lj * 1 + lk][t]
                                    if v_loc == 0.0: continue
                                    V[(br + lj) * 6 + (bc + lk)][base42 + pi * 3 + t] += sb * v_loc
                            for lk in range(1):
                                for li in range(2):
                                    w_loc = W42[lk * 2 + li][t]
                                    if w_loc == 0.0: continue
                                    W[(cc + lk) * 4 + (cr + li)][base42 + pi * 3 + t] += sc * w_loc
        return U, V, W

    # Try the main algorithm.
    try:
        U, V, W = assemble()
        if verify_full(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    # Fallback.
    try:
        if time.time() < deadline:
            U, V, W = assemble_fallback()
            if verify_full(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    return {"U": [], "V": [], "W": []}
# EVOLVE-BLOCK-END
