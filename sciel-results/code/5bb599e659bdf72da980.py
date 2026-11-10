import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Integer kissing configuration in dimension 11.

    Structural approach:
      * Candidate pool = all integer vectors of squared norm 6 in Z^11
        (exactly three coordinates in {+1,-1}, rest 0).
        Count = C(11,3) * 2^3 = 165 * 8 = 1320.
      * For equal-norm vectors with norm^2 = N, the contract condition
            min_pairwise_squared_distance >= max_squared_norm
        is equivalent (for distinct points) to  u . v <= N/2  for all pairs.
        With N = 6 this needs u . v <= 3, which is ALWAYS true for distinct
        norm-6 vectors (largest possible inner product is 3).  Hence ANY
        set of DISTINCT norm-6 vectors is valid, so we just maximize how many
        distinct candidates we can keep within the time budget.
      * Greedy add (randomized order, vectorized conflict test) + local
        search (add, rare remove) until a hard deadline.  Duplicates are
        prevented via a membership set.
      * Final validity is verified exactly with one vectorized Gram matrix.
    """
    start = time.time()
    DEADLINE = 600.0            # evaluator kills at 630s; keep a 30s margin
    d = 11
    N = 6
    half = N // 2               # inner-product upper bound for validity

    # ---- build candidate pool (norm^2 = 6 vectors) ----
    pos = np.array(list(itertools.combinations(range(d), 3)), dtype=int)
    signs = np.array(list(itertools.product([1, -1], repeat=3)), dtype=int)
    pool = np.zeros((len(pos) * len(signs), d), dtype=np.int64)
    for k, p in enumerate(pos):
        base = k * len(signs)
        for s in range(len(signs)):
            row = pool[base + s]
            row[p[0]] = signs[s, 0]
            row[p[1]] = signs[s, 1]
            row[p[2]] = signs[s, 2]
    n_cand = pool.shape[0]

    rng = np.random.default_rng(2024)

    # ---- greedy initialization in randomized order ----
    order = rng.permutation(n_cand)
    selected = []
    selected_set = set()
    sel_rows = []
    for c in order:
        c = int(c)
        if c in selected_set:
            continue
        v = pool[c]
        if sel_rows:
            S = np.array(sel_rows)
            if np.max(S @ v) > half:
                continue
        selected.append(c)
        selected_set.add(c)
        sel_rows.append(v)

    if len(selected) < 2:
        # fallback guaranteed-valid set (norm^2 = 2 vectors)
        pts = []
        for i in range(d):
            for j in range(i + 1, d):
                v = [0] * d
                v[i] = 1
                v[j] = 1
                pts.append(v)
        return np.array(pts, dtype=np.int64)

    selected = np.array(selected, dtype=int)
    S = pool[selected]
    k = len(selected)

    def valid_set(mat):
        if mat.shape[0] < 2:
            return False
        gram = mat @ mat.T
        np.fill_diagonal(gram, 0)
        return bool(np.max(gram) <= half)

    if not valid_set(S):
        return np.array(S, dtype=np.int64)

    def try_add(candidate_idx, mat, sel, selset):
        if candidate_idx in selset:
            return None
        v = pool[candidate_idx]
        if np.max(mat @ v) > half:
            return None
        new_sel = np.append(sel, candidate_idx)
        new_mat = np.vstack([mat, v.reshape(1, -1)])
        selset.add(candidate_idx)
        return new_mat, new_sel

    # ---- local search: keep adding while time allows ----
    probe_order = rng.permutation(n_cand)
    probe_pos = 0

    while time.time() < DEADLINE - 5.0:
        added = 0
        chunk_end = min(probe_pos + 4000, n_cand)
        while probe_pos < chunk_end:
            c = int(probe_order[probe_pos])
            probe_pos += 1
            res = try_add(c, S, selected, selected_set)
            if res is not None:
                S, selected = res
                k += 1
                added += 1
        if probe_pos >= n_cand:
            probe_pos = 0
            probe_order = rng.permutation(n_cand)
        # rare remove to escape a local optimum
        if added == 0 and k > 2 and rng.random() < 0.02:
            r = int(rng.integers(0, k))
            removed = int(selected[r])
            new_sel = np.delete(selected, r)
            new_mat = np.delete(S, r, axis=0)
            if valid_set(new_mat):
                S, selected = new_mat, new_sel
                selected_set.discard(removed)
                k -= 1

    # ---- final exact verification (vectorized) ----
    if not valid_set(S):
        pts = []
        for i in range(d):
            for j in range(i + 1, d):
                v = [0] * d
                v[i] = 1
                v[j] = 1
                pts.append(v)
        S = np.array(pts, dtype=np.int64)

    S = np.around(S).astype(np.int64)
    return S
