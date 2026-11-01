"""Matrix multiplication 2x3x4 rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 142  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8

    # Build the target tensor entries (the 24 ones)
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx))
    num_entries = len(tensor_entries)  # 24

    # Schoolbook fallback (rank 24)
    best_rank = n * m * p
    U_sb = [[0] * best_rank for _ in range(nm)]
    V_sb = [[0] * best_rank for _ in range(mp)]
    W_sb = [[0] * best_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}

    # ------------------------------------------------------------------
    # Exact integer verification (no float drift).  Work in doubled
    # integers: 2U, 2V, 2W are integers.  The tensor value is
    #   T = (1/8) * sum_r (2U)(2V)(2W)
    # so T == 1  <=>  sum_r (2U)(2V)(2W) == 8,  T == 0 <=> == 0.
    # ------------------------------------------------------------------
    def verify_exact(U, V, W, rank):
        U2 = np.array(U, dtype=np.int64) * 2
        V2 = np.array(V, dtype=np.int64) * 2
        W2 = np.array(W, dtype=np.int64) * 2
        for (u, v, w) in tensor_entries:
            s = 0
            for r in range(rank):
                s += U2[u, r] * V2[v, r] * W2[w, r]
            if s != 8:
                return False
        # Non-entries
        entry_set = set(tensor_entries)
        for u in range(nm):
            for v in range(mp):
                for w in range(pn):
                    if (u, v, w) in entry_set:
                        continue
                    s = 0
                    for r in range(rank):
                        s += U2[u, r] * V2[v, r] * W2[w, r]
                    if s != 0:
                        return False
        return True

    def snap_half(mat):
        return np.round(np.asarray(mat, dtype=np.float64) * 2.0) / 2.0

    # ------------------------------------------------------------------
    # Exact linear solve for one factor given the other two.
    #   sum_r A[idx, r] * X[x_idx, r] = b
    # Unknowns X flattened to (size * rank).  Returns snapped X or None.
    # ------------------------------------------------------------------
    def solve_factor(A, b, size, x_idx, rank):
        # Build design matrix rows only over the (size,rank) unknowns that
        # are actually coupled through x_idx.
        cols = np.array([x_idx * rank + r for r in range(rank)])  # (rank,)
        # A is (num_entries, rank); the coefficient for unknown (x_idx,r)
        # is A[:, r].  Build full (num_entries, size*rank) but sparse.
        M = np.zeros((num_entries, size * rank), dtype=np.float64)
        for r in range(rank):
            M[:, cols[r]] = A[:, r]
        # Solve least squares; accept only if residual ~ 0.
        try:
            sol, res, rk, sv = np.linalg.lstsq(M, b.astype(np.float64), rcond=None)
        except np.linalg.LinAlgError:
            return None
        # residual check
        r = M @ sol - b.astype(np.float64)
        if np.max(np.abs(r)) > 1e-7:
            return None
        X = sol.reshape(size, rank)
        Xs = snap_half(X)
        return Xs

    # ------------------------------------------------------------------
    # Alternating least-squares refinement over half-integers.
    # ------------------------------------------------------------------
    def als_refine(U, V, W, rank, max_iters=60):
        U = snap_half(U).astype(np.float64)
        V = snap_half(V).astype(np.float64)
        W = snap_half(W).astype(np.float64)
        b1 = np.ones(num_entries, dtype=np.float64)
        for it in range(max_iters):
            if time.time() > deadline - 4:
                break
            # ---- solve W (given U, V) ----
            Aw = np.zeros((num_entries, rank))
            for e, (u, v, w) in enumerate(tensor_entries):
                Aw[e, :] = U[u, :] * V[v, :]
            Wn = solve_factor(Aw, b1, pn, np.array([w for (_, _, w) in tensor_entries]), rank)
            if Wn is not None:
                W = Wn
            # ---- solve U (given V, W) ----
            Au = np.zeros((num_entries, rank))
            for e, (u, v, w) in enumerate(tensor_entries):
                Au[e, :] = V[v, :] * W[w, :]
            Un = solve_factor(Au, b1, nm, np.array([u for (u, _, _) in tensor_entries]), rank)
            if Un is not None:
                U = Un
            # ---- solve V (given U, W) ----
            Av = np.zeros((num_entries, rank))
            for e, (u, v, w) in enumerate(tensor_entries):
                Av[e, :] = U[u, :] * W[w, :]
            Vn = solve_factor(Av, b1, mp, np.array([v for (_, v, _) in tensor_entries]), rank)
            if Vn is not None:
                V = Vn
            # exact check
            if verify_exact(U, V, W, rank):
                return U, V, W
        return None

    # ------------------------------------------------------------------
    # Structured sparse basis generation (inspiration 2 idea):
    # build columns from support patterns x value patterns, with random
    # sign flips, so the random U,V are biased toward structures that
    # yield simple (half-integer) W.
    # ------------------------------------------------------------------
    def random_structured_matrix(size, rank, rng, support_choices, val_choices):
        M = np.zeros((size, rank), dtype=np.float64)
        for c in range(rank):
            sup = support_choices[rng.randrange(len(support_choices))]
            vals = val_choices[rng.randrange(len(val_choices))]
            # map support positions to values with random signs
            for pos, val in zip(sup, vals):
                if pos < size and val != 0:
                    M[pos, c] = val * (1.0 if rng.rand() < 0.5 else -1.0)
        return M

    # Precompute support/value pattern lists.
    # U size 6, V size 12, W size 8.
    def build_patterns(size):
        support = []
        # singletons
        for a in range(size):
            support.append([a])
        # pairs
        for a in range(size):
            for b in range(a + 1, size):
                support.append([a, b])
        # triples
        for a in range(size):
            for b in range(a + 1, size):
                for c in range(b + 1, size):
                    support.append([a, b, c])
        # value patterns consistent with support length
        vset = [0.0, 0.5, 1.0, 1.5, 2.0]
        val = []
        for s in support:
            L = len(s)
            # all same value
            for v in vset:
                if v != 0.0:
                    val.append([v] * L)
            # mixed small values
            if L >= 2:
                val.append([1.0, 0.5] + [0.5] * (L - 2))
                val.append([1.0, 1.0] + [0.5] * (L - 2))
                val.append([2.0, 0.5] + [0.5] * (L - 2))
        return support, val

    # We need support/val paired by length; simplest: regenerate per matrix
    # with a fixed small pattern pool keyed by length.
    def make_pattern_pool(size):
        pool = {}
        for L in (1, 2, 3):
            sups = []
            if L == 1:
                for a in range(size):
                    sups.append([a])
            elif L == 2:
                for a in range(size):
                    for b in range(a + 1, size):
                        sups.append([a, b])
            else:
                for a in range(size):
                    for b in range(a + 1, size):
                        for c in range(b + 1, size):
                            sups.append([a, b, c])
            vset = [0.5, 1.0, 1.5, 2.0]
            vals = []
            for v in vset:
                vals.append([v] * L)
            if L >= 2:
                vals.append([1.0, 0.5] + [0.5] * (L - 2))
                vals.append([1.0, 1.0] + [0.5] * (L - 2))
                vals.append([2.0, 0.5] + [0.5] * (L - 2))
            pool[L] = (sups, vals)
        return pool

    poolU = make_pattern_pool(nm)
    poolV = make_pattern_pool(mp)
    poolW = make_pattern_pool(pn)

    def gen_structured(pool, size, rank, rng):
        M = np.zeros((size, rank), dtype=np.float64)
        for c in range(rank):
            L = rng.choice([1, 2, 3])
            sups, vals = pool[L]
            sup = sups[rng.randrange(len(sups))]
            vals_c = vals[rng.randrange(len(vals))]
            for pos, val in zip(sup, vals_c):
                if pos < size:
                    M[pos, c] = val * (1.0 if rng.rand() < 0.5 else -1.0)
        return M

    # ------------------------------------------------------------------
    # Main search over ranks.
    # ------------------------------------------------------------------
    b1 = np.ones(num_entries, dtype=np.float64)
    w_idx_arr = np.array([w for (_, _, w) in tensor_entries])
    u_idx_arr = np.array([u for (u, _, _) in tensor_entries])
    v_idx_arr = np.array([v for (_, v, _) in tensor_entries])

    for target_rank in range(18, best_rank):
        if time.time() > deadline - 8:
            break

        found = False
        # number of restarts scales with remaining budget
        base_attempts = 4000
        for attempt in range(base_attempts):
            if time.time() > deadline - 4:
                break

            seed = 1000 + target_rank * 17 + attempt
            rng = np.random.RandomState(seed)

            # Structured random U, V
            U = gen_structured(poolU, nm, target_rank, rng)
            V = gen_structured(poolV, mp, target_rank, rng)

            # Solve W exactly
            Aw = np.zeros((num_entries, target_rank))
            for e, (u, v, w) in enumerate(tensor_entries):
                Aw[e, :] = U[u, :] * V[v, :]
            W = solve_factor(Aw, b1, pn, w_idx_arr, target_rank)
            if W is None:
                continue

            # Quick exact check
            if verify_exact(U, V, W, target_rank):
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                found = True
                break

            # ALS refinement to fix U and V
            res = als_refine(U, V, W, target_rank, max_iters=40)
            if res is not None:
                U, V, W = res
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                found = True
                break

        if found:
            break

    return best_result
# EVOLVE-BLOCK-END
