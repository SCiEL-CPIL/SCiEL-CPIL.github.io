import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21
    NTRIPLES = n * m * p  # 105

    # ------------------------------------------------------------------
    # Column representation: each bilinear form is a triple (u, v, w)
    # ------------------------------------------------------------------
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
            return (u, v, w)

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
        return (u, v, w)

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
    # Build the tensor matrix M (NTRIPLES x len(pool))
    # ------------------------------------------------------------------
    def build_M(pool):
        r = len(pool)
        M = [[0] * r for _ in range(NTRIPLES)]
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
        if time.time() > deadline - 5.0:
            return list(pool)
        M = build_M(pool)
        piv = exact_pivots(M, deadline - 5.0)
        piv_set = set(piv)
        keep = [c for c in range(len(pool)) if c not in piv_set]
        reduced = [pool[c] for c in keep]
        if verify_pool(reduced):
            return reduced
        return list(pool)

    # ------------------------------------------------------------------
    # Generate all possible 2x2x2 Strassen blocks
    # ------------------------------------------------------------------
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))

    # Build conflict graph
    num_blocks = len(all_blocks)
    conflict = [[False] * num_blocks for _ in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if all_blocks[i][1] & all_blocks[j][1]:
                conflict[i][j] = True
                conflict[j][i] = True

    # Backtracking search for maximum independent set
    best_set = []
    
    def backtrack(start, current_set):
        nonlocal best_set
        if time.time() > deadline - 30.0:
            return
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]
        
        # Pruning
        if len(current_set) + (num_blocks - start) <= len(best_set):
            return

        for i in range(start, num_blocks):
            # Check conflict with all in current_set
            if any(conflict[i][idx] for idx in current_set):
                continue
                
            current_set.append(i)
            backtrack(i + 1, current_set)
            current_set.pop()

    # Start backtracking
    backtrack(0, [])
    
    best_packing = [all_blocks[i][0] for i in best_set]

    # ------------------------------------------------------------------
    # Build the pool from best packing + schoolbook for uncovered
    # ------------------------------------------------------------------
    pool = []
    covered = set()
    for block_params in best_packing:
        i1, i2, j1, j2, k1, k2 = block_params
        pool.extend(build_strassen_cols(i1, i2, j1, j2, k1, k2))
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    pool.append(build_schoolbook_col(i, j, k))

    # ------------------------------------------------------------------
    # Try to reduce the pool using exact elimination
    # ------------------------------------------------------------------
    reduced_pool = reduce_pool(pool, deadline)
    
    # Use the reduced pool if it's smaller and valid
    if len(reduced_pool) < len(pool):
        final_pool = reduced_pool
    else:
        final_pool = pool

    # ------------------------------------------------------------------
    # Convert to numpy matrices
    # ------------------------------------------------------------------
    rank = len(final_pool)
    U = np.zeros((U_rows, rank), dtype=float)
    V = np.zeros((V_rows, rank), dtype=float)
    W = np.zeros((W_rows, rank), dtype=float)

    for c, (u, v, w) in enumerate(final_pool):
        for r in range(U_rows):
            U[r, c] = u[r]
        for r in range(V_rows):
            V[r, c] = v[r]
        for r in range(W_rows):
            W[r, c] = w[r]

    # Verify correctness
    T = np.zeros((U_rows, V_rows, W_rows))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.sum((T_comp - T) ** 2)

    if error > 1e-9:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((U_rows, rank), dtype=float)
        V = np.zeros((V_rows, rank), dtype=float)
        W = np.zeros((W_rows, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
