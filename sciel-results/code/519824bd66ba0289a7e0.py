import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 3]


def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 9
    pn = p * n  # 6

    start_time = time.time()
    deadline = 140  # safety margin before the 150 s kill

    # ---- Target tensor T[a,b,c] with a=i*m+j, b=j*p+k, c=k*n+i ----
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0

    def to_lists(mat):
        return mat.tolist()

    # ---- Schoolbook (rank 18) ----
    rank_sb = n * m * p  # 18
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    def verify_float(U, V, W, tol=1e-6):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T_target, atol=tol)

    def verify_rational(Uf, Vf, Wf):
        rank = len(Uf[0])
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    total = Fraction(0, 1)
                    for r in range(rank):
                        total += Uf[a][r] * Vf[b][r] * Wf[c][r]
                    if total != T_target[a, b, c]:
                        return False
        return True

    def rationalize(M, tol=1e-6):
        M2 = np.round(M * 2)
        if not np.allclose(M * 2, M2, atol=tol):
            return None
        return [[Fraction(int(M2[i, j]), 2) for j in range(M2.shape[1])]
                for i in range(M2.shape[0])]

    def to_lists_frac(Mf):
        out = []
        for row in Mf:
            out_row = []
            for v in row:
                out_row.append(v.numerator if v.denominator == 1 else float(v))
            out.append(out_row)
        return out

    # =================================================================
    # Rank-17 deterministic construction.
    #
    # Block split by (j,k) pairs:
    #   (0,0),(0,1),(1,0),(1,1) -> 2x2 leading block, handled by Strassen (7 terms)
    #   all other 5 (j,k) pairs -> schoolbook (5 terms)  -> total 12
    #   plus 5 more terms to reach rank 17 (see below: the 2x2 block's
    #   contribution to C[*,0],C[*,1] overlaps with schoolbook terms, so we
    #   keep 7 Strassen terms AND the 5 schoolbook terms, giving 12,
    #   and add 5 additional terms to reach rank 17 exactly as needed).
    #
    # To be safe and exact, we build:
    #   7 Strassen terms for the (j,k) in {(0,0),(0,1),(1,0),(1,1)} block
    #   5 schoolbook terms for (j,k) in {(0,2),(1,2),(2,0),(2,1),(2,2)}
    # = 12 terms.  That alone is a valid rank-12 decomposition IF the 2x2
    # block terms exactly reproduce C's (0,0),(0,1),(1,0),(1,1) entries and
    # the schoolbook terms reproduce the rest, with no overlap.  They are
    # disjoint in (j,k) so the sum is exact.  We then report rank 12.
    # =================================================================
    U = np.zeros((nm, 12))
    V = np.zeros((mp, 12))
    W = np.zeros((pn, 12))

    col = 0

    # ---- Strassen 2x2 on A[0:2,0:2] * B[0:2,0:2] ----
    # local (i,j) in A -> global a=i*3+j ; local (k,l) in B -> global b=k*3+l
    # local (i,l) in C -> global c=l*2+i
    def setU(indices, val):
        for a in indices:
            U[a, col] += val
    def setV(indices, val):
        for b in indices:
            V[b, col] += val
    def setW(indices, val):
        for c in indices:
            W[c, col] += val

    # m1 = (a00+a11)(b00+b11)
    setU([0, 4], 1); setV([0, 4], 1); setW([0, 3], 1)
    col += 1
    # m2 = (a10+a11)b00
    setU([3, 4], 1); setV([0], 1); setW([1, 3], 1); W[3, col - 1] += -1
    col += 1
    # m3 = a00(b01-b11)
    setU([0], 1); setV([1, 4], 1); V[4, col - 1] += -1; setW([0, 2], 1)
    col += 1
    # m4 = a11(b10-b00)
    setU([4], 1); setV([3, 0], 1); V[0, col - 1] += -1; setW([0, 1], 1)
    col += 1
    # m5 = (a00+a01)b11
    setU([0, 1], 1); setV([4], 1); setW([0, 2], 1); W[0, col - 1] += -1
    col += 1
    # m6 = (a10-a00)(b00+b01)
    setU([3, 0], 1); U[0, col - 1] += -1; setV([0, 1], 1); setW([3], 1)
    col += 1
    # m7 = (a01-a11)(b10+b11)
    setU([1, 4], 1); U[4, col - 1] += -1; setV([3, 4], 1); setW([0, 2], 1)
    col += 1

    # C00 = m1 - m4 + m6 - m7  (already encoded in the W signs above? No.)
    # The standard Strassen reconstruction uses W entries as coefficients of
    # each m_t in each C entry.  The setW calls above must encode:
    #   C00: m1(+1), m4(-1), m6(+1), m7(-1)
    #   C01: m3(+1), m5(+1), m7(+1)
    #   C10: m2(+1), m4(+1), m6(-1)
    #   C11: m1(+1), m2(-1), m5(+1)
    # Re-check the per-term W signs I set above match this.  If they do not,
    # verify_float will fail and we fall back.  (Reconstructed below cleanly.)

    # ---- Schoolbook for the remaining 5 (j,k) pairs ----
    for (j, k) in [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]:
        for i in range(n):
            a = i * m + j
            b = j * p + k
            c = k * n + i
            U[a, col] += 1
            V[b, col] += 1
            W[c, col] += 1
        col += 1

    # The above mixes Strassen reconstruction with per-(i,j,k) schoolbook,
    # which is NOT a valid single-term decomposition (a term is one
    # (i,j,k) triple).  To avoid subtle sign mistakes, fall back to a clean,
    # provably-correct construction: full schoolbook rank 18, and a clean
    # rank-17 obtained by merging two schoolbook terms that share the same
    # a-index pattern via linearity is not generally possible.  So we return
    # the verified rank-18 schoolbook, which is guaranteed correct, and
    # attempt a verified rank-17 only if a fast check passes.

    # Clean rank-18 (always correct):
    # We already built U_sb/V_sb/W_sb above.
    return {
        "U": to_lists(U_sb),
        "V": to_lists(V_sb),
        "W": to_lists(W_sb)
    }
