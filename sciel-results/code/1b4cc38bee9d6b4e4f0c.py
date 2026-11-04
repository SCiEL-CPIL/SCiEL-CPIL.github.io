"""Matrix multiplication 3x5x7 - rank-minimal exact tensor decomposition search.
Combines disjoint Strassen-block packing (current) with exact column-elimination
rank reduction (inspiration) to squeeze out additional rank savings."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21
    NTRIPLES = n * m * p  # 105

    # ------------------------------------------------------------------
    # Generate all possible 2x2x2 blocks
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
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), frozenset(indices)))

    num_blocks = len(all_blocks)

    # Precompute conflict graph
    block_conflicts = [set() for _ in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if all_blocks[i][1] & all_blocks[j][1]:
                block_conflicts[i].add(j)
                block_conflicts[j].add(i)

    # ------------------------------------------------------------------
    # Greedy packing with multiple orderings
    # ------------------------------------------------------------------
    best_num_blocks = 0
    best_packing = []

    random.seed(42)

    # Deterministic orders
    base_orders = [
        list(range(num_blocks)),
        list(reversed(range(num_blocks))),
        sorted(range(num_blocks), key=lambda x: (all_blocks[x][0][2], all_blocks[x][0][4], all_blocks[x][0][0])),
        sorted(range(num_blocks), key=lambda x: (all_blocks[x][0][4], all_blocks[x][0][2], all_blocks[x][0][0])),
        sorted(range(num_blocks), key=lambda x: len(block_conflicts[x])),
        sorted(range(num_blocks), key=lambda x: -len(block_conflicts[x])),
    ]

    for order in base_orders:
        used_indices = set()
        current_packing = []
        for idx in order:
            block_set = all_blocks[idx][1]
            if not (block_set & used_indices):
                current_packing.append(idx)
                used_indices |= block_set
        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing.copy()

    # Randomized greedy trials
    num_trials = 3000
    for t in range(num_trials):
        if time.time() > deadline - 40.0:
            break
        indices = list(range(num_blocks))
        random.shuffle(indices)
        used_indices = set()
        current_packing = []
        for idx in indices:
            block_set = all_blocks[idx][1]
            if not (block_set & used_indices):
                current_packing.append(idx)
                used_indices |= block_set
        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing.copy()
            if best_num_blocks >= 12:
                break

    # ------------------------------------------------------------------
    # Local search to improve packing
    # ------------------------------------------------------------------
    current_packing = best_packing.copy()
    improved = True
    max_local_iters = 50
    local_iter = 0

    while improved and local_iter < max_local_iters and time.time() < deadline - 20.0:
        improved = False
        local_iter += 1

        current_set = set(current_packing)

        # 1. Try adding a block
        for idx in range(num_blocks):
            if idx in current_set:
                continue
            conflict = False
            for p_idx in current_packing:
                if idx in block_conflicts[p_idx]:
                    conflict = True
                    break
            if not conflict:
                current_packing.append(idx)
                current_set.add(idx)
                improved = True
                break

        if improved:
            continue

        # 2. Try removing one block and adding two
        for remove_idx in current_packing:
            if time.time() > deadline - 20.0:
                break

            trial_packing = [x for x in current_packing if x != remove_idx]
            trial_set = set(trial_packing)

            candidates = []
            for add_idx in range(num_blocks):
                if add_idx in trial_set or add_idx == remove_idx:
                    continue
                ok = True
                for p_idx in trial_packing:
                    if add_idx in block_conflicts[p_idx]:
                        ok = False
                        break
                if ok:
                    candidates.append(add_idx)

            found_pair = False
            for i_c in range(len(candidates)):
                for j_c in range(i_c + 1, len(candidates)):
                    c1 = candidates[i_c]
                    c2 = candidates[j_c]
                    if c1 not in block_conflicts[c2]:
                        current_packing = trial_packing + [c1, c2]
                        improved = True
                        found_pair = True
                        break
                if found_pair:
                    break

            if found_pair:
                break

        if len(current_packing) > best_num_blocks:
            best_num_blocks = len(current_packing)
            best_packing = current_packing.copy()
            if best_num_blocks >= 13:
                break

    # ------------------------------------------------------------------
    # Build column representations
    # ------------------------------------------------------------------
    def build_strassen_cols(i1, i2, j1, j2, k1, k2):
        cols = []
        a11, a12 = i1 * m + j1, i1 * m + j2
        a21, a22 = i2 * m + j1, i2 * m + j2
        b11, b12 = j1 * p + k1, j1 * p + k2
        b21, b22 = j2 * p + k1, j2 * p + k2
        c11, c12 = k1 * n + i1, k2 * n + i1
        c21, c22 = k1 * n + i2, k2 * n + i2

        def T(ud, vd, wd):
            u = [0] * U_rows
            v = [0] * V_rows
            w = [0] * W_rows
            for idx, val in ud.items():
                u[idx] = val
            for idx, val in vd.items():
                v[idx] = val
            for idx, val in wd.items():
                w[idx] = val
            return (u, v, w)

        cols.append(T({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1}))
        cols.append(T({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1}))
        cols.append(T({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1}))
        cols.append(T({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1}))
        cols.append(T({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1}))
        cols.append(T({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1}))
        cols.append(T({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1}))
        return cols

    def build_schoolbook_col(i, j, k):
        u = [0] * U_rows
        v = [0] * V_rows
        w = [0] * W_rows
        u[i * m + j] = 1
        v[j * p + k] = 1
        w[k * n + i] = 1
        return (u, v, w)

    # Build the full pool of columns
    pool = []
    covered = set()
    for idx in best_packing:
        i1, i2, j1, j2, k1, k2 = all_blocks[idx][0]
        pool.extend(build_strassen_cols(i1, i2, j1, j2, k1, k2))
        covered.update(all_blocks[idx][1])

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    pool.append(build_schoolbook_col(i, j, k))

    # ------------------------------------------------------------------
    # Exact verification
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

    if not verify_pool(pool):
        # Fallback: pure schoolbook
        pool = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    pool.append(build_schoolbook_col(i, j, k))

    # ------------------------------------------------------------------
    # Column elimination: find linearly dependent columns and remove them
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

    def exact_pivots(M, dl):
        rows = len(M)
        cols = len(M[0])
        A = [list(row) for row in M]
        pivot_cols = []
        cur = 0
        for c in range(cols):
            if time.time() > dl:
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

    def reduce_pool(pool, dl):
        if len(pool) <= 1:
            return pool
        M = build_M(pool)
        piv = exact_pivots(M, dl)
        piv_set = set(piv)
        keep = [c for c in range(len(pool)) if c not in piv_set]
        reduced = [pool[c] for c in keep]
        if verify_pool(reduced):
            return reduced
        return list(pool)

    # Apply reduction
    if time.time() < deadline - 15.0:
        pool = reduce_pool(pool, deadline - 15.0)

    # ------------------------------------------------------------------
    # Convert pool to U, V, W matrices
    # ------------------------------------------------------------------
    rank = len(pool)
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]

    for c in range(rank):
        u, v, w = pool[c]
        for r in range(U_rows):
            U[r][c] = u[r]
        for r in range(V_rows):
            V[r][c] = v[r]
        for r in range(W_rows):
            W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}
