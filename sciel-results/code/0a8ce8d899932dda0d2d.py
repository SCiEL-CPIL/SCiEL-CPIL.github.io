"""Matrix multiplication 3,5,7 - rank-minimal exact tensor decomposition.
Combines disjoint Strassen-block packing with row-column decoupling (m=2+3)
and local search for rank reduction."""
import time
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 148.0

    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21
    NTRIPLES = n * m * p  # 105

    # ------------------------------------------------------------------
    # Column representation: each bilinear form is a triple (u, v, w)
    # of integer vectors of lengths U_rows, V_rows, W_rows.
    # A pool of columns is "exact" iff for every (i,j,k):
    #   sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] == 1
    # ------------------------------------------------------------------
    def triple_to_cols(u, v, w):
        return (list(u), list(v), list(w))

    def build_strassen_cols(i1, i2, j1, j2, k1, k2):
        """Return the 7 Strassen bilinear-form columns for a 2x2x2 block."""
        cols = []
        a11, a12 = i1 * m + j1, i1 * m + j2
        a21, a22 = i2 * m + j1, i2 * m + j2
        b11, b12 = j1 * p + k1, j1 * p + k2
        b21, b22 = j2 * p + k1, j2 * p + k2
        c11, c12 = k1 * n + i1, k2 * n + i1
        c21, c22 = k1 * n + i2, k2 * n + i2

        def T(ud, vd, wd):
            u = [0] * U_rows; v = [0] * V_rows; w = [0] * W_rows
            for idx, val in ud.items(): u[idx] = val
            for idx, val in vd.items(): v[idx] = val
            for idx, val in wd.items(): w[idx] = val
            return triple_to_cols(u, v, w)

        # M1 = (a11+a22)*(b11+b22) -> c11 + c22
        cols.append(T({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1}))
        # M2 = (a21+a22)*b11 -> c21 - c22
        cols.append(T({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1}))
        # M3 = a11*(b12-b22) -> c12 + c22
        cols.append(T({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1}))
        # M4 = a22*(b21-b11) -> c11 + c21
        cols.append(T({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1}))
        # M5 = (a11+a12)*b22 -> -c11 + c12
        cols.append(T({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1}))
        # M6 = (a21-a11)*(b11+b12) -> c22
        cols.append(T({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1}))
        # M7 = (a12-a22)*(b21+b22) -> c11
        cols.append(T({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1}))
        return cols

    def build_schoolbook_col(i, j, k):
        u = [0] * U_rows; v = [0] * V_rows; w = [0] * W_rows
        u[i * m + j] = 1
        v[j * p + k] = 1
        w[k * n + i] = 1
        return triple_to_cols(u, v, w)

    # ------------------------------------------------------------------
    # Verify exactness of a pool of columns (Fraction arithmetic).
    # ------------------------------------------------------------------
    def verify_pool(pool):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    for u, v, w in pool:
                        uv = u[ui] * v[vj]
                        if uv:
                            total += uv * w[wk]
                    if total != 1:
                        return False
        return True

    # ------------------------------------------------------------------
    # Build the tensor matrix M (NTRIPLES x len(pool)) where
    # M[(i,j,k), c] = u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i].
    # The pool is exact iff M is all-ones; rank(M) is the minimal rank.
    # ------------------------------------------------------------------
    def build_M(pool):
        r = len(pool)
        M = [ [0] * r for _ in range(NTRIPLES) ]
        idx = 0
        for i in range(n):
            ui = i * m
            for j in range(m):
                vj = j * p
                for k in range(p):
                    wk = k * n + i
                    row = M[idx]
                    for c in range(r):
                        u, v, w = pool[c]
                        val = u[ui + j] * v[vj + k] * w[wk]
                        row[c] = val
                    idx += 1
        return M

    # ------------------------------------------------------------------
    # Exact Gaussian elimination over Fraction; returns pivot columns.
    # ------------------------------------------------------------------
    def exact_pivots(M, deadline):
        rows = len(M)
        cols = len(M[0])
        A = [list(row) for row in M]
        pivot_cols = []
        cur = 0
        for c in range(cols):
            if time.time() > deadline:
                break
            if cur >= rows:
                break
            piv = -1
            for r in range(cur, rows):
                if A[r][c] != 0:
                    piv = r
                    break
            if piv == -1:
                continue
            A[cur], A[piv] = A[piv], A[cur]
            pv = A[cur][c]
            for r in range(cur + 1, rows):
                if A[r][c] != 0:
                    f = A[r][c] / pv
                    for cc in range(c, cols):
                        A[r][cc] -= f * A[cur][cc]
            pivot_cols.append(c)
            cur += 1
        return pivot_cols

    # ------------------------------------------------------------------
    # Reduce a pool to a minimal-rank exact subset using pivots.
    # ------------------------------------------------------------------
    def reduce_pool(pool, deadline):
        M = build_M(pool)
        piv = exact_pivots(M, deadline)
        piv_set = set(piv)
        keep = [c for c in range(len(pool)) if c not in piv_set]
        reduced = [pool[c] for c in keep]
        if verify_pool(reduced):
            return reduced
        # Fallback: keep all (should still be exact)
        return list(pool)

    # ------------------------------------------------------------------
    # Local search: try removing one column at a time if still exact.
    # ------------------------------------------------------------------
    def local_search_reduce(pool, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            new_pool = list(pool)
            for c in range(len(new_pool)):
                if time.time() > deadline:
                    break
                # Try removing column c
                candidate = new_pool[:c] + new_pool[c+1:]
                if verify_pool(candidate):
                    new_pool = candidate
                    improved = True
                    break
        return new_pool

    # ------------------------------------------------------------------
    # Greedy packing of disjoint 2x2x2 Strassen blocks (multiple orders).
    # ------------------------------------------------------------------
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            idxs = set()
                            for ii in (i1, i2):
                                for jj in (j1, j2):
                                    for kk in (k1, k2):
                                        idxs.add((ii, jj, kk))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), idxs))

    def greedy_packing(order):
        used = set()
        packing = []
        for params, idxs in order:
            if not (idxs & used):
                packing.append(params)
                used |= idxs
        return packing

    # Deterministic packings from several fixed orders (fast, no RNG needed).
    packings = []
    base_orders = [
        all_blocks,
        list(reversed(all_blocks)),
        sorted(all_blocks, key=lambda b: (b[0][2], b[0][4], b[0][0])),
        sorted(all_blocks, key=lambda b: (b[0][4], b[0][2], b[0][0])),
    ]
    for order in base_orders:
        packings.append(greedy_packing(order))

    # A few randomized orders if time permits.
    import random
    rng = random.Random(12345)
    for _ in range(300):
        if time.time() > deadline - 60.0:
            break
        shuffled = list(all_blocks)
        rng.shuffle(shuffled)
        packings.append(greedy_packing(shuffled))

    # ------------------------------------------------------------------
    # Build pools from each packing, reduce, keep the best.
    # ------------------------------------------------------------------
    best_pool = None
    best_rank = None

    for packing in packings:
        if time.time() > deadline - 40.0:
            break
        pool = []
        covered = set()
        for params in packing:
            pool.extend(build_strassen_cols(*params))
            i1, i2, j1, j2, k1, k2 = params
            for ii in (i1, i2):
                for jj in (j1, j2):
                    for kk in (k1, k2):
                        covered.add((ii, jj, kk))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        pool.append(build_schoolbook_col(i, j, k))

        if not verify_pool(pool):
            continue

        reduced = reduce_pool(pool, deadline - 30.0)
        # Apply local search for further reduction
        reduced = local_search_reduce(reduced, deadline - 20.0)
        r = len(reduced)
        if best_rank is None or r < best_rank:
            best_rank = r
            best_pool = reduced

    # ------------------------------------------------------------------
    # Row-column decoupling: partition m=5 into m1=2, m2=3
    # Build independent decompositions for 3x2x7 and 3x3x7 sub-tensors
    # ------------------------------------------------------------------
    if time.time() < deadline - 30.0:
        m1, m2 = 2, 3  # Partition of m=5
        
        # Sub-tensor 1: 3 x 2 x 7 (rows 0..1 in m-dimension)
        # Sub-tensor 2: 3 x 3 x 7 (rows 2..4 in m-dimension)
        
        def build_subtensor_pool(n_sub, m_sub, p_sub, offset_i, offset_j):
            """Build a pool for an n_sub x m_sub x p_sub sub-tensor."""
            pool = []
            # Try all possible Strassen blocks within this sub-tensor
            sub_blocks = []
            for i1 in range(n_sub - 1):
                for i2 in range(i1 + 1, n_sub):
                    for j1 in range(m_sub - 1):
                        for j2 in range(j1 + 1, m_sub):
                            for k1 in range(p_sub - 1):
                                for k2 in range(k1 + 1, p_sub):
                                    idxs = set()
                                    for ii in (i1, i2):
                                        for jj in (j1, j2):
                                            for kk in (k1, k2):
                                                idxs.add((ii, jj, kk))
                                    sub_blocks.append(((i1, i2, j1, j2, k1, k2), idxs))
            
            # Greedy packing
            used = set()
            packing = []
            for params, idxs in sub_blocks:
                if not (idxs & used):
                    packing.append(params)
                    used |= idxs
            
            for params in packing:
                i1, i2, j1, j2, k1, k2 = params
                # Map to global indices
                gi1, gi2 = i1 + offset_i, i2 + offset_i
                gj1, gj2 = j1 + offset_j, j2 + offset_j
                cols = build_strassen_cols(gi1, gi2, gj1, gj2, k1, k2)
                pool.extend(cols)
            
            # Cover remaining triples with schoolbook
            covered = set()
            for params in packing:
                i1, i2, j1, j2, k1, k2 = params
                gi1, gi2 = i1 + offset_i, i2 + offset_i
                gj1, gj2 = j1 + offset_j, j2 + offset_j
                for ii in (gi1, gi2):
                    for jj in (gj1, gj2):
                        for kk in (k1, k2):
                            covered.add((ii, jj, kk))
            
            for i in range(n_sub):
                for j in range(m_sub):
                    for k in range(p_sub):
                        gi, gj = i + offset_i, j + offset_j
                        if (gi, gj, k) not in covered:
                            pool.append(build_schoolbook_col(gi, gj, k))
            
            return pool
        
        # Build sub-pools
        pool1 = build_subtensor_pool(n, m1, p, 0, 0)  # 3x2x7, m-offset=0
        pool2 = build_subtensor_pool(n, m2, p, 0, m1)  # 3x3x7, m-offset=2
        
        combined_pool = pool1 + pool2
        
        if verify_pool(combined_pool):
            reduced = reduce_pool(combined_pool, deadline - 25.0)
            reduced = local_search_reduce(reduced, deadline - 15.0)
            r = len(reduced)
            if best_rank is None or r < best_rank:
                best_rank = r
                best_pool = reduced
        
        # Try alternative partition: m1=3, m2=2
        if time.time() < deadline - 25.0:
            m1_alt, m2_alt = 3, 2
            pool1_alt = build_subtensor_pool(n, m1_alt, p, 0, 0)  # 3x3x7, m-offset=0
            pool2_alt = build_subtensor_pool(n, m2_alt, p, 0, m1_alt)  # 3x2x7, m-offset=3
            
            combined_pool_alt = pool1_alt + pool2_alt
            
            if verify_pool(combined_pool_alt):
                reduced = reduce_pool(combined_pool_alt, deadline - 20.0)
                reduced = local_search_reduce(reduced, deadline - 10.0)
                r = len(reduced)
                if best_rank is None or r < best_rank:
                    best_rank = r
                    best_pool = reduced

    # ------------------------------------------------------------------
    # Fallback: pure schoolbook (always exact, rank = 105).
    # ------------------------------------------------------------------
    if best_pool is None:
        best_pool = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best_pool.append(build_schoolbook_col(i, j, k))
        best_rank = len(best_pool)

    # ------------------------------------------------------------------
    # Assemble U, V, W matrices (rows x rank) from the reduced pool.
    # ------------------------------------------------------------------
    rank = len(best_pool)
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    for c, (u, v, w) in enumerate(best_pool):
        for r in range(U_rows):
            U[r][c] = u[r]
        for r in range(V_rows):
            V[r][c] = v[r]
        for r in range(W_rows):
            W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
