"""Matrix multiplication 3x5x6 rank-minimal tensor decomposition.

Combines:
  * block partitioning with recursive Strassen (2x2x2 -> rank 7) blocks
  * exhaustive search over all 1/2 groupings of each dimension
  * a linear-dependency elimination pass on the schoolbook basis tensors
    (probe for a rank-89 exact decomposition)

The entry point construct() returns {'U','V','W'} with the lowest rank found.
"""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin before 120 s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    # ---- tensor index helpers --------------------------------------------
    # tensor T_{i,j,k} lives at (Urow, Vrow, Wrow)
    def urow(i, j):
        return i * m + j
    def vrow(j, k):
        return j * p + k
    def wrow(k, i):
        return k * n + i

    # ---- verification (numpy, fast) --------------------------------------
    def verify(U, V, W, rank):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        u_vals = Ua[i_idx * m + j_idx, :]
        v_vals = Va[j_idx * p + k_idx, :]
        w_vals = Wa[k_idx * n + i_idx, :]
        result = np.sum(u_vals * v_vals * w_vals, axis=1)
        return bool(np.all(np.abs(result - 1.0) < 1e-9))

    # ---- Strassen 2x2x2 terms -------------------------------------------
    def strassen_terms():
        return [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,1): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (1,0): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]

    STERMS = strassen_terms()

    # ---- block decomposition builder -------------------------------------
    def build_block_decomposition(i_groups, j_groups, k_groups):
        total_rank = 0
        block_info = []
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    r = 7 if (si == 2 and sj == 2 and sk == 2) else si * sj * sk
                    total_rank += r
                    block_info.append((ig, jg, kg, r))

        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]

        col = 0
        for ig, jg, kg, r in block_info:
            si, sj, sk = len(ig), len(jg), len(kg)
            if si == 2 and sj == 2 and sk == 2:
                for u_pat, v_pat, w_pat in STERMS:
                    for (li, lj), val_u in u_pat.items():
                        U[urow(ig[li], jg[lj])][col] = val_u
                    for (lj, lk), val_v in v_pat.items():
                        V[vrow(jg[lj], kg[lk])][col] = val_v
                    for (lk, li), val_w in w_pat.items():
                        W[wrow(kg[lk], ig[li])][col] = val_w
                    col += 1
            else:
                for li in range(si):
                    for lj in range(sj):
                        for lk in range(sk):
                            gi, gj, gk = ig[li], jg[lj], kg[lk]
                            U[urow(gi, gj)][col] = 1
                            V[vrow(gj, gk)][col] = 1
                            W[wrow(gk, gi)][col] = 1
                            col += 1
        return {"U": U, "V": V, "W": W}

    # ---- schoolbook (rank 90) --------------------------------------------
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[urow(i, j)][col] = 1
                    V[vrow(j, k)][col] = 1
                    W[wrow(k, i)][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # ---- linear-dependency elimination on schoolbook tensors -------------
    def try_linear_dependency_elimination():
        """
        Each schoolbook term is a rank-1 tensor. A set of terms with
        coefficients c_t is a zero-tensor iff for every (i,j,k):
            sum_t c_t * u_t[i,j] * v_t[j,k] * w_t[k,i] == 0.
        Build a linear system in c_t and look for a non-trivial solution.
        If found, the corresponding term can be dropped (rank-89) while the
        sum of all terms still equals the multiplication tensor.
        """
        # enumerate terms
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append((i, j, k))
        T = len(terms)  # 90

        # For each (i,j,k) constraint, build the coefficient vector.
        # Only terms with the same (i,j,k) contribute (u,v,w are 0/1 deltas),
        # so each constraint row is sparse with a single 1 at its term index.
        # We need at least one constraint per term to detect a dependency;
        # here each (i,j,k) constrains exactly its own term => identity system,
        # which has no non-trivial null vector. So a pure single-term drop is
        # impossible; but we still build the general system and check the null
        # space to be safe (it may reveal structure we missed).
        constraints = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = np.zeros(T)
                    for t, (ti, tj, tk) in enumerate(terms):
                        if ti == i and tj == j and tk == k:
                            row[t] = 1.0
                    constraints.append(row)
        A = np.array(constraints, dtype=float)
        # null space
        try:
            _, s, vh = np.linalg.svd(A)
        except Exception:
            return None
        tol = 1e-10 * max(s)
        null_rank = int(np.sum(s > tol))
        null_dim = T - null_rank
        if null_dim <= 0:
            return None
        # pick the null vector with the largest-magnitude single coefficient
        null_vecs = vh[null_rank:, :]
        best = None
        for nv in null_vecs:
            idx = int(np.argmax(np.abs(nv)))
            mag = abs(nv[idx])
            if mag < 1e-6:
                continue
            if best is None or mag > best[1]:
                best = (idx, mag)
        if best is None:
            return None
        drop_idx = best[0]
        # drop that term
        keep = [t for t in range(T) if t != drop_idx]
        rank = len(keep)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for t in keep:
            i, j, k = terms[t]
            U[urow(i, j)][col] = 1
            V[vrow(j, k)][col] = 1
            W[wrow(k, i)][col] = 1
            col += 1
        return {"U": U, "V": V, "W": W}

    # ---- partition generation --------------------------------------------
    def generate_partitions(dim):
        partitions = []
        def backtrack(start, current):
            if start == dim:
                partitions.append(list(current))
                return
            if start + 1 <= dim:
                current.append([start])
                backtrack(start + 1, current)
                current.pop()
            if start + 2 <= dim:
                current.append([start, start + 1])
                backtrack(start + 2, current)
                current.pop()
        backtrack(0, [])
        return partitions

    i_partitions = generate_partitions(n)
    j_partitions = generate_partitions(m)
    k_partitions = generate_partitions(p)

    # ---- search ----------------------------------------------------------
    best_result = schoolbook()
    best_rank = n * m * p

    # 1) exhaustive partition search (Strassen blocks)
    for ip in i_partitions:
        for jp in j_partitions:
            for kp in k_partitions:
                if time.time() > deadline:
                    break
                result = build_block_decomposition(ip, jp, kp)
                rank = len(result["U"][0])
                if rank < best_rank:
                    if verify(result["U"], result["V"], result["W"], rank):
                        best_rank = rank
                        best_result = result
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # 2) linear-dependency elimination probe (only if it can beat best)
    if time.time() < deadline - 5:
        ld = try_linear_dependency_elimination()
        if ld is not None:
            ld_rank = len(ld["U"][0])
            if ld_rank < best_rank:
                if verify(ld["U"], ld["V"], ld["W"], ld_rank):
                    best_rank = ld_rank
                    best_result = ld

    return best_result
# EVOLVE-BLOCK-END
