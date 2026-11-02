import numpy as np
import itertools
import time

D = 11

def _greedy(order, compat, N):
    # Max-min style greedy: pick first, then repeatedly pick the unselected
    # vertex (in `order`) with the most already-selected neighbours.
    selected = np.zeros(N, dtype=bool)
    count = 0
    for idx in order:
        if selected[idx]:
            continue
        if count == 0:
            selected[idx] = True
            count += 1
        else:
            c = int(np.sum(compat[idx] & selected))
            if c > 0:
                selected[idx] = True
                count += 1
    return selected, count

def construct():
    start = time.time()
    deadline = start + 143.0

    # ---- Build candidate pool (all squared norm = 4) ----
    cands = []
    # Type 1: one 2, rest 0  -> 22
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            cands.append(v)
    # Type 2: four 1s, rest 0  -> C(11,4)*16 = 3360
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            cands.append(v)
    # Type 3: two 2s, rest 0  -> C(11,2)*4 = 220
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                cands.append(v)

    A = np.array(cands, dtype=np.int32)
    N = len(A)

    M = A @ A.T
    compat = (M <= 2).astype(np.uint8)
    np.fill_diagonal(compat, 0)

    rng = np.random.default_rng(12345)
    best_set = None
    best_count = 0

    # ---- Phase 1: many greedy restarts ----
    t1_budget = deadline - 40.0
    trials = 0
    # deterministic first
    sel, c = _greedy(np.arange(N), compat, N)
    if c > best_count:
        best_count, best_set = c, sel.copy()
    while time.time() < t1_budget:
        trials += 1
        # random permutation, but a cheap randomised order (partial) is fine too
        order = rng.permutation(N)
        sel, c = _greedy(order, compat, N)
        if c > best_count:
            best_count, best_set = c, sel.copy()
            if best_count >= 150:  # strong; keep a bit but don't over-invest
                pass

    # ---- Phase 2: local search (add + swap) ----
    if best_set is None:
        best_set = np.zeros(N, dtype=bool)
    # Phase 2a: add any compatible unselected vector
    unselected = np.where(~best_set)[0]
    for idx in unselected:
        if time.time() > deadline - 8:
            break
        if np.any(compat[idx] & best_set):
            continue
        best_set[idx] = True
        best_count += 1

    # Phase 2b: swap-based local search
    max_iter = 200
    it = 0
    while it < max_iter and time.time() < deadline - 6:
        it += 1
        sel_idx = np.where(best_set)[0]
        uns_idx = np.where(~best_set)[0]
        if len(sel_idx) < 3 or len(uns_idx) == 0:
            break
        improved = False
        # remove up to 3 random selected, then greedily re-add
        n_rem = min(3, len(sel_idx))
        rem = rng.choice(len(sel_idx), size=n_rem, replace=False)
        rem_set = sel_idx[rem]
        test = best_set.copy()
        for r in rem_set:
            test[r] = False
        test_count = best_count - n_rem
        # greedily add from unselected, prefer most neighbours (max-min)
        add_pool = np.where(~test)[0]
        # order by current number of neighbours in test (descending)
        neigh = np.array([int(np.sum(compat[u] & test)) for u in add_pool])
        order_add = add_pool[np.argsort(-neigh, kind='stable')]
        for a in order_add:
            if time.time() > deadline - 6:
                break
            if test[a]:
                continue
            if np.any(compat[a] & test):
                test[a] = True
                test_count += 1
        if test_count > best_count:
            best_set = test
            best_count = test_count
            improved = True
        if not improved and it > 60:
            break

    if best_set is None or best_count == 0:
        out = []
        for i in range(D):
            for s in (1, -1):
                v = [0] * D
                v[i] = 2 * s
                out.append(v)
        return np.array(out, dtype=np.int64)

    return A[best_set].astype(np.int64)
