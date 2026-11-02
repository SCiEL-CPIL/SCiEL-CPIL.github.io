import time
import numpy as np

DIMENSIONS = [3, 4, 8]
DEADLINE = time.time() + 118.0  # safety margin before the 150s wall-clock kill


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8

    # Correct 7-term Strassen 2x2x2 (M1..M7).  Local index = i*2 + j for A,
    # j*2 + k for B, k*2 + i for C.  Output:
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    strassen_decomp = [
        ({0: 1, 3: 1},   {0: 1, 3: 1},   {0: 1, 3: 1}),    # M1
        ({2: 1, 3: 1},   {0: 1},         {2: 1, 3: -1}),   # M2
        ({0: 1},         {1: 1, 3: -1},  {1: 1, 3: 1}),    # M3
        ({3: 1},         {2: 1, 0: -1},  {0: 1, 2: 1}),    # M4
        ({0: 1, 1: 1},   {3: 1},         {1: 1, 0: -1}),   # M5
        ({2: 1, 0: -1},  {0: 1, 1: 1},   {3: 1}),          # M6
        ({1: 1, 3: -1},  {2: 1, 3: 1},   {0: 1}),          # M7
    ]

    def _build_block_strassen():
        """Rank-88 block decomposition: n=3 -> 2+1, m=4 -> 2+2, p=8 -> 4x2.
        Top (rows 0,1): Strassen per 2x2x2 block.  Bottom (row 2): schoolbook."""
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0

        for j1 in range(2):          # two m-blocks of size 2
            # top part: rows 0,1 -> 2x2x8, split p into 4 blocks of 2
            for l in range(4):
                for u_c, v_c, w_c in strassen_decomp:
                    for li, cf in u_c.items():
                        li_i, li_j = li // 2, li % 2
                        U[(li_i) * m + (2 * j1 + li_j)][col] += cf
                    for li, cf in v_c.items():
                        li_i, li_j = li // 2, li % 2
                        V[(2 * j1 + li_i) * p + (2 * l + li_j)][col] += cf
                    for li, cf in w_c.items():
                        li_i, li_j = li // 2, li % 2
                        W[(2 * l + li_j) * n + (li_i)][col] += cf
                    col += 1
            # bottom part: row 2 -> schoolbook 1x2x8 = 16
            for k in range(8):
                for j2 in range(2):
                    gj = 2 * j1 + j2
                    U[2 * m + gj][col] = 1
                    V[gj * p + k][col] = 1
                    W[k * n + 2][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _build_alt_split():
        """Alt: n=3 -> 1+2.  Top row 0 schoolbook (32); bottom rows 1,2 Strassen (56)."""
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for j in range(4):
            for k in range(8):
                U[j][col] = 1
                V[j * p + k][col] = 1
                W[k * n][col] = 1
                col += 1
        for j1 in range(2):
            for l in range(4):
                for u_c, v_c, w_c in strassen_decomp:
                    for li, cf in u_c.items():
                        li_i, li_j = li // 2, li % 2
                        U[(1 + li_i) * m + (2 * j1 + li_j)][col] += cf
                    for li, cf in v_c.items():
                        li_i, li_j = li // 2, li % 2
                        V[(2 * j1 + li_i) * p + (2 * l + li_j)][col] += cf
                    for li, cf in w_c.items():
                        li_i, li_j = li // 2, li % 2
                        W[(2 * l + li_j) * n + (1 + li_i)][col] += cf
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _verify(U, V, W, tol=1e-9):
        r = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i * m + j] * Va[j * p + k], Wa[k * n + i])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def _rank_of(mat):
        return np.linalg.matrix_rank(np.array(mat, dtype=float), tol=1e-9)

    def _reduce(U, V, W, budget):
        """Greedy: drop columns that are linearly dependent on the others AND whose
        removal keeps the tensor equal to 1.  Uses a cheap tensor check so we only
        spend a full matrix_rank when a column actually looks removable.
        
        Strategy: 
        1. First, identify columns that are exact duplicates (same U, V, W vectors)
        2. Then, try to merge pairs of columns that are linearly dependent
        3. Finally, do a full greedy pass
        """
        r = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        keep = list(range(r))

        # tensor values for the current (full) decomposition
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = np.dot(Ua[i * m + j] * Va[j * p + k], Wa[k * n + i])

        def tensor_ok(idx, tol=1e-9):
            # remove column idx from the tensor
            t = T.copy()
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        t[i, j, k] -= Ua[i * m + j, idx] * Va[j * p + k, idx] * Wa[k * n + i, idx]
            return bool(np.allclose(t, 1.0, atol=tol))

        def is_redundant(idx, keep_set, tol=1e-9):
            """Check if column idx is redundant given the set of kept columns."""
            if idx not in keep_set:
                return False
            # fast tensor check first
            if not tensor_ok(idx):
                return False
            # then confirm the column is a linear combination of the rest
            cols = [c for c in keep_set if c != idx]
            if not cols:
                return False
            if _rank_of(np.column_stack([Ua[:, idx], Ua[:, cols]])) == \
               _rank_of(Ua[:, cols]):
                if _rank_of(np.column_stack([Va[:, idx], Va[:, cols]])) == \
                   _rank_of(Va[:, cols]):
                    if _rank_of(np.column_stack([Wa[:, idx], Wa[:, cols]])) == \
                       _rank_of(Wa[:, cols]):
                        return True
            return False

        # Phase 1: Remove exact duplicates
        # Group columns by their (U, V, W) vector triple
        from collections import defaultdict
        col_groups = defaultdict(list)
        for idx in range(r):
            # Create a hashable key from the column vectors
            u_key = tuple(Ua[:, idx])
            v_key = tuple(Va[:, idx])
            w_key = tuple(Wa[:, idx])
            col_groups[(u_key, v_key, w_key)].append(idx)
        
        for group in col_groups.values():
            if len(group) > 1 and time.time() < budget:
                # Keep only the first column in each duplicate group
                for dup_idx in group[1:]:
                    if is_redundant(dup_idx, set(keep)):
                        keep = [c for c in keep if c != dup_idx]

        # Phase 2: Greedy removal of linearly dependent columns
        # Try multiple passes to find more reductions
        for _pass in range(3):
            if time.time() > budget:
                break
            progress = False
            for idx in range(r):
                if time.time() > budget:
                    break
                if idx not in keep:
                    continue
                if is_redundant(idx, set(keep)):
                    keep = [c for c in keep if c != idx]
                    progress = True
            if not progress:
                break

        U_red = [[U[i][c] for c in keep] for i in range(len(U))]
        V_red = [[V[i][c] for c in keep] for i in range(len(V))]
        W_red = [[W[i][c] for c in keep] for i in range(len(W))]
        return U_red, V_red, W_red, len(keep)

    # --- build candidate solutions and keep the best (lowest verified rank) ---
    best = None
    best_rank = 10 ** 9

    for builder in (_build_block_strassen, _build_alt_split):
        if time.time() > DEADLINE:
            break
        sol = builder()
        U, V, W = sol["U"], sol["V"], sol["W"]
        if not _verify(U, V, W):
            continue
        # polish: greedy linear-dependency reduction
        U2, V2, W2, rk = _reduce(U, V, W, DEADLINE)
        if rk < best_rank and _verify(U2, V2, W2):
            best = {"U": U2, "V": V2, "W": W2}
            best_rank = rk
        # also keep the un-reduced one if it is better and verified
        if len(U[0]) < best_rank and _verify(U, V, W):
            best = {"U": U, "V": V, "W": W}
            best_rank = len(U[0])

    # fallback: naive rank-96 (always valid)
    if best is None:
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
        best = {"U": U, "V": V, "W": W}
        best_rank = rank

    return best
