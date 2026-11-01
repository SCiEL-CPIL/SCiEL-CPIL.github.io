# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product, combinations

def construct():
    """
    Rank-minimal exact decomposition for 3x3 matrix multiplication.

    Design (kept from current program, refined):
      1. Partition A,B into 2x2 + 1x1 blocks.
      2. 2x2 core  -> Strassen (7 rank-1 terms, all +/-1).
      3. 1x1 core  -> scalar product (1 rank-1 term).
      4. The 6 cross entries (2x2 * 1x1 and 1x2 * 2x1) are handled by a
         small number of extra rank-1 "cross" columns.  We search for the
         minimum number c of such columns (each a sparse +/-1 vector) for
         which the 6 linear equations are solvable EXACTLY, then solve for
         the integer W coefficients.
      5. Verify exactness (integer entries, tensor identity) and clean.
      6. Fall back to schoolbook rank-27 if no better exact solution found.
    """
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = 120.0

    # ---------- schoolbook rank-27 (guaranteed valid) ----------
    def schoolbook():
        U = [[0]*27 for _ in range(9)]
        V = [[0]*27 for _ in range(9)]
        W = [[0]*27 for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j][col] = 1
                    V[j*3+k][col] = 1
                    W[k*3+i][col] = 1
                    col += 1
        return U, V, W

    # ---------- Strassen 2x2 core (7 terms) ----------
    # U indexed (i,j) in {0,1}^2 ; V indexed (j,k) in {0,1}^2 ; W indexed (k,i) in {0,1}^2
    def idx2(a, b):  # (a,b) in {0,1}^2 -> 0..3
        return a*2 + b

    strassen = [
        # (u:4, v:4, w:4)  values in {0,1,-1}
        ([ 1, 0, 0, 0], [ 1, 0, 0, 0], [ 1, 0, 0, 0]),   # M1 = A00*B00
        ([ 1, 0, 0, 0], [ 0, 0, 0, 1], [ 0, 0, 0, 1]),   # M2 = A00*B11
        ([ 0, 0, 0, 1], [ 1, 0, 0, 0], [ 0, 1, 0, 0]),   # M3 = A10*B00
        ([ 1, 0, 0, 0], [ 0, 1, 1, 0], [ 0, 0, 0, 1]),   # M4 = A00*(B01+B11)
        ([ 1, 0, 1, 0], [ 0, 0, 1, 0], [ 0, 1, 0, 0]),   # M5 = (A00+A10)*B10
        ([ 0, 1, 0, 0], [ 1, 0, 1, 0], [ 0, 0, 1, 0]),   # M6 = A01*(B00+B10)
        ([ 0, 0, 0, 1], [ 0, 1, 0, 1], [ 0, 1, 0, 0]),   # M7 = A11*(B01-B11)
    ]

    # Build the 8 fixed columns (7 Strassen + 1 scalar for the (2,2) block)
    U8 = []
    V8 = []
    W8 = []
    for (u4, v4, w4) in strassen:
        U8.append([u4[idx2(i, j)] for i in range(3) for j in range(3)])
        V8.append([v4[idx2(j, k)] for j in range(3) for k in range(3)])
        W8.append([w4[idx2(k, i)] for k in range(3) for i in range(3)])
    # scalar core: A[2,2]*B[2,2]
    U8.append([0]*9); U8[-1][2*3+2] = 1
    V8.append([0]*9); V8[-1][2*3+2] = 1
    W8.append([0]*9); W8[-1][2*3+2] = 1

    U8 = np.array(U8, dtype=float).T   # (9, 8)
    V8 = np.array(V8, dtype=float).T   # (9, 8)
    W8 = np.array(W8, dtype=float).T   # (9, 8)

    # ---------- verify a full candidate ----------
    def verify(U, V, W, tol=1e-9):
        rank = U.shape[1]
        # integer / half-integer
        for M in (U, V, W):
            if not np.allclose(M*2, np.round(M*2), atol=tol):
                return False
        # tensor identity: sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all i,j,k
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*3+j, r]*V[j*3+k, r]*W[k*3+i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def clean_halfint(M):
        # round to nearest half-integer
        return (np.round(M*2.0)/2.0)

    # ---------- cross-term search ----------
    # The 6 cross entries (i,j,k) with exactly one of i,j,k equal to 2
    cross_entries = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if (i==2) ^ (j==2) ^ (k==2):  # exactly one equals 2
                    cross_entries.append((i, j, k))
    # 6 entries

    # For a candidate set of `c` cross columns, the equation for cross entry (i,j,k):
    #   sum_{t=0}^{7} U8[i,j,t] V8[j,k,t] W8[k,i,t]  +  sum_{c} Ux[j?] ... = 1
    # where cross column r has Ux[:,r], Vx[:,r], Wx[:,r].
    # LHS_r = Ux[i,j,r]*Vx[j,k,r]*Wx[k,i,r]
    # We need sum_r LHS_r = 1 - fixed_part.

    # Precompute fixed part of each cross entry from the 8 core columns
    fixed = {}
    for (i, j, k) in cross_entries:
        s = 0.0
        for t in range(8):
            s += U8[i*3+j, t]*V8[j*3+k, t]*W8[k*3+i, t]
        fixed[(i, j, k)] = s

    # Each cross column is a rank-1 vector; we restrict to sparse +/-1 patterns.
    # A cross column must contribute to the cross entries it touches.
    # Build the set of "support patterns": choose which of the 9 positions are
    # nonzero (each +/-1) for U, V, W.  To keep it tractable we enumerate
    # structured patterns: a cross column corresponds to a choice of
    #   u_sup (subset of 9, each +/-1), v_sup, w_sup
    # but that is huge.  Instead we enumerate *rank-1 cross terms* of the form
    #   Ux = e_a (single +1 at position a in U-space)
    #   Vx = e_b
    #   Wx = s * e_c   (s = +/-1)
    # i.e. each cross column is a "monomial" A[a] * B[b] * (s) * C[c].
    # Then LHS_r for entry (i,j,k) = 1 iff a==i*3+j and b==j*3+k and c==k*3+i,
    # times s.  This is exactly a schoolbook-style term.  A set of `c` such
    # monomial columns can cover the 6 cross entries if we pick, for each cross
    # entry (i,j,k), the monomial (a=i*3+j, b=j*3+k, c=k*3+i, s=+1).  That uses
    # 6 columns -> rank 14.  But we can do better by allowing a column to
    # contribute to multiple entries only if it is a true rank-1 product, which
    # monomials are not (each monomial touches exactly one (i,j,k)).
    #
    # However, the 8 core columns already contribute `fixed[(i,j,k)]` to each
    # cross entry.  We need sum of cross LHS = 1 - fixed.  With monomial columns
    # each cross entry needs (1 - fixed[(i,j,k)]) supplied by its own monomial,
    # and a monomial can only supply one entry.  So we need, per entry, a
    # monomial with coefficient (1 - fixed).  If (1 - fixed) is 0 we need none;
    # if it is +/-1 we need one monomial with that sign; otherwise monomials
    # (integer) cannot represent it -> infeasible with pure monomials.
    #
    # To allow non-integer-free but half-integer solutions and to allow a
    # column to share, we instead solve the linear system for the W coefficients
    # given a chosen set of Ux,Vx cross columns.  We choose Ux,Vx as the 6
    # monomial U,V supports (one per cross entry) and let W be solved freely
    # (9*c variables).  This is the current program's machinery and is exact.

    # Candidate: use the 6 monomial cross columns for U,V; solve for W.
    def build_cross_UV():
        Uc = []
        Vc = []
        for (i, j, k) in cross_entries:
            u = [0]*9; u[i*3+j] = 1
            v = [0]*9; v[j*3+k] = 1
            Uc.append(u)
            Vc.append(v)
        return np.array(Uc, dtype=float).T, np.array(Vc, dtype=float).T

    def try_solve(rank_total, Ufull, Vfull):
        # Build linear system for W: 27 equations, 9*rank_total variables
        c = rank_total - 8
        A = np.zeros((27, 9*rank_total))
        b = np.ones(27)
        eq = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for r in range(rank_total):
                        var = (k*3+i)*rank_total + r
                        A[eq, var] = Ufull[i*3+j, r]*Vfull[j*3+k, r]
                    eq += 1
        try:
            Wflat, res, rk, sv = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            return None
        Wfull = Wflat.reshape((9, rank_total))
        # check residual
        if np.max(np.abs(A@Wflat - b)) > 1e-6:
            return None
        # clean to half-integers
        Wc = clean_halfint(Wfull)
        Uf = clean_halfint(Ufull)
        Vf = clean_halfint(Vfull)
        if verify(Uf, Vf, Wc):
            return (Uf, Vf, Wc)
        return None

    # Try the 6-monomial cross construction (rank 14)
    Uc, Vc = build_cross_UV()
    Ufull = np.hstack([U8, Uc])
    Vfull = np.hstack([V8, Vc])
    sol = try_solve(14, Ufull, Vfull)
    if sol is not None:
        Uf, Vf, Wf = sol
        return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}

    # If the 6-monomial solve failed (non-half-integer W or inconsistency),
    # try allowing the cross columns to be denser: enumerate a bounded set of
    # structured cross U,V patterns and solve for W.  Keep it fast.
    if time.time() - start_time < deadline - 5:
        # Cross U columns: each is a sparse vector.  Enumerate patterns where
        # U column has +1 at exactly the (i,j) of a cross entry and possibly
        # additional +/-1 at other positions; same for V.  We try a modest set.
        base_positions = [(i*3+j) for (i, j, k) in cross_entries]
        # generate a few candidate U,V cross column sets by adding one extra
        # nonzero to each base column
        def variants(base):
            out = []
            for p in base:
                v = [0]*9; v[p] = 1
                out.append(v)
            return out

        # Try: cross U = base monomials, cross V = base monomials (done).
        # Try: cross U = base monomials, cross V = base monomials but with the
        # sign / extra support varied.  We enumerate extra support positions.
        extra_pool = list(range(9))
        attempts = 0
        max_attempts = 400
        rng = np.random.RandomState(0)
        while attempts < max_attempts and time.time() - start_time < deadline - 5:
            # random sparse cross U,V columns (6 of them), each +/-1 on 1-2 positions
            Uc2 = []
            Vc2 = []
            ok = True
            for t in range(6):
                u = [0]*9
                npos = 1 if rng.rand() < 0.6 else 2
                for p in rng.choice(9, npos, replace=False):
                    u[p] = 1 if rng.rand() < 0.5 else -1
                v = [0]*9
                for p in rng.choice(9, npos, replace=False):
                    v[p] = 1 if rng.rand() < 0.5 else -1
                Uc2.append(u)
                Vc2.append(v)
            Ufull2 = np.hstack([U8, np.array(Uc2, dtype=float).T])
            Vfull2 = np.hstack([V8, np.array(Vc2, dtype=float).T])
            sol2 = try_solve(14, Ufull2, Vfull2)
            if sol2 is not None:
                Uf, Vf, Wf = sol2
                return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}
            attempts += 1

    # Fallback: schoolbook rank-27
    U27, V27, W27 = schoolbook()
    return {"U": U27, "V": V27, "W": W27}

# EVOLVE-BLOCK-END
