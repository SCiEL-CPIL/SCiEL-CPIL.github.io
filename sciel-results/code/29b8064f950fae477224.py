"""
Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.

Strategy:
  1. Build a structured rank-44 base: Strassen for (2,2,2) sub-blocks + schoolbook.
  2. Search for linear dependencies to reduce rank (duplicate cols, pairwise, random).
  3. Try multiple structured decompositions with different splittings.
  4. Validate with exact rational arithmetic.
  5. Return the lowest-rank valid decomposition found.
"""
import time
import sys
import random
from fractions import Fraction

def construct():
    t0 = time.time()
    deadline = t0 + 135.0

    def _ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8
    num_terms = n * m * p  # 48

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
        U = candidate["U"]
        V = candidate["V"]
        W = candidate["W"]
        rank = len(U[0])
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    acc = Fraction(0)
                    for c in range(rank):
                        uv = U[i * m + a][c]
                        if uv == 0:
                            continue
                        vv = V[a * p + k][c]
                        if vv == 0:
                            continue
                        ww = W[k * n + i][c]
                        if ww == 0:
                            continue
                        acc += Fraction(uv) * Fraction(vv) * Fraction(ww)
                    if acc != 1:
                        return False
        return True

    # Strassen 2x2x2 rank-7
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

    # Verify Strassen
    strassen_ok = True
    for i in range(2):
        for a in range(2):
            for k in range(2):
                acc = 0
                for r in range(7):
                    acc += S_U[r][i*2+a] * S_V[r][a*2+k] * S_W[r][k*2+i]
                if acc != 1:
                    strassen_ok = False
                    break
            if not strassen_ok:
                break
        if not strassen_ok:
            break

    if not strassen_ok:
        return _schoolbook()

    def _build_split(a_split, b_split):
        """Build decomposition with given split.
        a_split: list of row-pair counts for A (n=2 rows split)
        Actually for 2x3*3x8:
        - A is 2x3, B is 3x8
        - We can split columns of A / rows of B (m-dimension): e.g., [2,1]
        - And columns of B (p-dimension): e.g., [4,4], [2,2,2,2], [8]
        """
        # m_split: how to partition the 3 middle dimension
        # p_split: how to partition the 8 output columns
        U = [[0]*num_terms for _ in range(n*m)]  # placeholder, will resize
        # We'll build column-by-column
        cols_u = []
        cols_v = []
        cols_w = []

        for mi, m_sz in enumerate(m_split):
            for pi, p_sz in enumerate(p_split):
                # Block: A[:, mi*m_sz:(mi+1)*m_sz] (2 x m_sz) * B[mi*m_sz:(mi+1)*m_sz, pi*p_sz:(pi+1)*p_sz] (m_sz x p_sz)
                # Result: 2 x p_sz
                if m_sz == 2 and p_sz == 2:
                    # Use Strassen
                    for r in range(7):
                        cu = [0]*(n*m)
                        cv = [0]*(m*p)
                        cw = [0]*(p*n)
                        for li in range(2):
                            for la in range(2):
                                val = S_U[r][li*2+la]
                                if val:
                                    cu[li*m + (mi*2+la)] = val
                        for la in range(2):
                            for lk in range(2):
                                val = S_V[r][la*2+lk]
                                if val:
                                    cv[(mi*2+la)*p + (pi*2+lk)] = val
                        for lk in range(2):
                            for li in range(2):
                                val = S_W[r][lk*2+li]
                                if val:
                                    cw[(pi*2+lk)*n + li] = val
                        cols_u.append(cu)
                        cols_v.append(cv)
                        cols_w.append(cw)
                else:
                    # Schoolbook for this block: 2 * m_sz * p_sz terms
                    for i in range(n):
                        for la in range(m_sz):
                            for lk in range(p_sz):
                                cu = [0]*(n*m)
                                cv = [0]*(m*p)
                                cw = [0]*(p*n)
                                cu[i*m + (mi*m_sz+la)] = 1
                                cv[(mi*m_sz+la)*p + (pi*p_sz+lk)] = 1
                                cw[(pi*p_sz+lk)*n + i] = 1
                                cols_u.append(cu)
                                cols_v.append(cv)
                                cols_w.append(cw)

        rank = len(cols_u)
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for c in range(rank):
            for i in range(n*m):
                U[i][c] = cols_u[c][i]
            for i in range(m*p):
                V[i][c] = cols_v[c][i]
            for i in range(p*n):
                W[i][c] = cols_w[c][i]
        return {"U": U, "V": V, "W": W}

    def _col_vec(U, V, W, c):
        vec = [0] * num_terms
        for i in range(n):
            for a in range(m):
                uv = U[i*m+a][c]
                if not uv:
                    continue
                for k in range(p):
                    vv = V[a*p+k][c]
                    if not vv:
                        continue
                    ww = W[k*n+i][c]
                    if not ww:
                        continue
                    vec[i*m*p + a*p + k] = uv * vv * ww
        return vec

    def _reduce_deps(candidate):
        """Try to reduce rank by finding linear dependencies among column vectors."""
        U, V, W = candidate["U"], candidate["V"], candidate["W"]
        rank = len(U[0])
        cols = [_col_vec(U, V, W, c) for c in range(rank)]

        # Check all columns sum correctly
        total = [0]*num_terms
        for cv in cols:
            for idx in range(num_terms):
                total[idx] += cv[idx]
        for idx in range(num_terms):
            if total[idx] != 1:
                return candidate, rank

        # Try removing one column at a time (check if it's needed)
        # A column can be removed if it's a linear combination of others that we can absorb
        # Simple approach: try all pairs that are negatives
        for i in range(rank):
            if not _ok():
                break
            for j in range(i+1, rank):
                is_neg = True
                for idx in range(num_terms):
                    if cols[i][idx] + cols[j][idx] != 0:
                        is_neg = False
                        break
                if is_neg:
                    # cols[i] = -cols[j], drop one
                    keep = [c for c in range(rank) if c != j]
                    nr = rank - 1
                    nU = [[0]*nr for _ in range(n*m)]
                    nV = [[0]*nr for _ in range(m*p)]
                    nW = [[0]*nr for _ in range(p*n)]
                    nc = 0
                    for c in keep:
                        for ii in range(n*m): nU[ii][nc] = U[ii][c]
                        for ii in range(m*p): nV[ii][nc] = V[ii][c]
                        for ii in range(p*n): nW[ii][nc] = W[ii][c]
                        nc += 1
                    cand = {"U": nU, "V": nV, "W": nW}
                    if _validate(cand):
                        return cand, nr

        # Try random small coefficient dependencies
        random.seed(12345)
        for trial in range(20000):
            if not _ok():
                break
            size = random.choice([3, 4, 5])
            if size > rank:
                size = rank
            indices = random.sample(range(rank), size)
            coeffs = [random.choice([-3,-2,-1,1,2,3]) for _ in range(size)]
            dep = True
            for idx in range(num_terms):
                s = sum(ci * cols[cc][idx] for ci, cc in zip(coeffs, indices))
                if s != 0:
                    dep = False
                    break
            if dep:
                drop = indices[-1]
                keep = [c for c in range(rank) if c != drop]
                nr = rank - 1
                nU = [[0]*nr for _ in range(n*m)]
                nV = [[0]*nr for _ in range(m*p)]
                nW = [[0]*nr for _ in range(p*n)]
                nc = 0
                for c in keep:
                    for ii in range(n*m): nU[ii][nc] = U[ii][c]
                    for ii in range(m*p): nV[ii][nc] = V[ii][c]
                    for ii in range(p*n): nW[ii][nc] = W[ii][c]
                    nc += 1
                cand = {"U": nU, "V": nV, "W": nW}
                if _validate(cand):
                    return cand, nr

        return candidate, rank

    # Generate candidate splittings
    candidates = []

    # Standard: m_split=[2,1], p_split=[2,2,2,2] → 4 Strassen blocks + 2x1x8 schoolbook = 28+16=44
    c1 = _build_split([2,1], [2,2,2,2])
    if _validate(c1):
        candidates.append(c1)

    # m_split=[2,1], p_split=[4,4] → 2 Strassen (2x2x4) + 1 schoolbook (2x1x8)
    # 2x2x4: for each 2x2 * 2x4 block = 2 Strassen blocks of 2x2x2 = 14
    # 2x1x4: schoolbook = 8
    # Total: 14+8=22 per m-split group... wait, let me think more carefully.
    # Actually _build_split handles it: m_split=[2,1], p_split=[4,4]
    # m_sz=2,p_sz=4: schoolbook (2*2*4=16) for the 2x2 part, schoolbook for 2x1 part (2*1*4=8)
    # Hmm, my function only uses Strassen when m_sz==2 AND p_sz==2.
    # Let me add Strassen for m_sz=2, p_sz even by decomposing p_sz into 2s.
    
    # Let me redo with a more flexible builder
    def _build_flex(m_split, p_split):
        cols_u, cols_v, cols_w = [], [], []
        for mi, m_sz in enumerate(m_split):
            for pi, p_sz in enumerate(p_split):
                if m_sz == 2 and p_sz % 2 == 0:
                    # Use Strassen on each 2x2 sub-block
                    for sub in range(p_sz // 2):
                        k_base = pi * p_sz + sub * 2
                        for r in range(7):
                            cu = [0]*(n*m)
                            cv = [0]*(m*p)
                            cw = [0]*(p*n)
                            for li in range(2):
                                for la in range(2):
                                    val = S_U[r][li*2+la]
                                    if val:
                                        cu[li*m + (mi*m_sz+la)] = val
                            for la in range(2):
                                for lk in range(2):
                                    val = S_V[r][la*2+lk]
                                    if val:
                                        cv[(mi*m_sz+la)*p + (k_base+lk)] = val
                            for lk in range(2):
                                for li in range(2):
                                    val = S_W[r][lk*2+li]
                                    if val:
                                        cw[(k_base+lk)*n + li] = val
                            cols_u.append(cu)
                            cols_v.append(cv)
                            cols_w.append(cw)
                else:
                    for i in range(n):
                        for la in range(m_sz):
                            for lk in range(p_sz):
                                cu = [0]*(n*m)
                                cv = [0]*(m*p)
                                cw = [0]*(p*n)
                                cu[i*m + (mi*m_sz+la)] = 1
                                cv[(mi*m_sz+la)*p + (pi*p_sz+lk)] = 1
                                cw[(pi*p_sz+lk)*n + i] = 1
                                cols_u.append(cu)
                                cols_v.append(cv)
                                cols_w.append(cw)
        rank = len(cols_u)
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for c in range(rank):
            for i in range(n*m): U[i][c] = cols_u[c][i]
            for i in range(m*p): V[i][c] = cols_v[c][i]
            for i in range(p*n): W[i][c] = cols_w[c][i]
        return {"U": U, "V": V, "W": W}

    if not candidates:
        c1 = _build_flex([2,1], [2,2,2,2])
        if _validate(c1):
            candidates.append(c1)

    # Try various splittings
    splitting_options = [
        ([2,1], [2,2,2,2]),   # 44
        ([2,1], [4,4]),       # 2*(14) + 16 = 44? Let's see: m=2,p=4 → 2 Strassen=14; m=1,p=4 → 8; total per p-block=22, ×2 p-blocks=44
        ([2,1], [8]),         # m=2,p=8 → 4 Strassen=28; m=1,p=8 → 16; total=44
        ([1,1,1], [2,2,2,2]), # all schoolbook: 1*1*2*4=8 per group... 4 groups × 8 = 32? No: 3 m-blocks × 4 p-blocks × 1*1*2=2 = 24. Wait: n=2, so 2*1*2=4 per (m_block,p_block). 3×4×4=48.
        ([1,2], [2,2,2,2]),   # m=1: 4×4=16; m=2: 4 Strassen×2=28; total=44
        ([3], [2,2,2,2]),     # m=3,p=2: schoolbook 2*3*2=12 per block, 4 blocks=48
        ([3], [4,4]),         # 2*3*4=24 per block, 2 blocks=48
        ([2,1], [3,5]),       # m=2,p=3: schoolbook 2*2*3=12; m=1,p=3: 2*1*3=6; per group=18. Second: m=2,p=5: 2*2*5=20; m=1,p=5:10; per group=30. Total=18+30=48.
    ]

    for m_sp, p_sp in splitting_options:
        if not _ok():
            break
        try:
            cand = _build_flex(m_sp, p_sp)
            if _validate(cand):
                candidates.append(cand)
        except Exception:
            pass

    # Also try schoolbook as fallback
    candidates.append(_schoolbook())

    # Pick best (lowest rank) and try to reduce
    best = candidates[0]
    best_rank = len(best["U"][0])

    for cand in candidates:
        r = len(cand["U"][0])
        if r < best_rank:
            best = cand
            best_rank = r

    # Try dependency reduction on the best
    if _ok():
        best, best_rank = _reduce_deps(best)

    # If still at 44, try reducing on other candidates too
    if best_rank >= 44:
        for cand in candidates:
            if not _ok():
                break
            r = len(cand["U"][0])
            if r >= best_rank:
                continue
            reduced, new_r = _reduce_deps(cand)
            if new_r < best_rank:
                best = reduced
                best_rank = new_r

    return best
