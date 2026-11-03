import numpy as np
import itertools
import time

D = 11
N_SUP = 330          # C(11,4)
N_T1 = 22            # 2 * 11 type-1 vectors

def construct():
    start = time.time()
    deadline = start + 142.0   # safety margin under 150s kill

    # --- Build supports (each = sorted 4-tuple of coordinate indices) ---
    supports = list(itertools.combinations(range(D), 4))
    sup = np.array(supports, dtype=np.int64)          # (N_SUP, 4)

    # --- Vectorized support-level conflict matrix ---
    # Two supports conflict iff they share >= 3 coordinates.
    # shared = C(4,3) * C(7,0) + C(4,4) * C(6,0) = 4 + 1 = 5
    a = sup[:, :, None]                 # (N,4,1)
    b = sup[None, :, :]                 # (1,N,4)
    eq = (a == b)                       # (N,N,4)
    shared = eq.sum(axis=2)             # (N,N)
    conflict = (shared >= 3)            # (N,N) bool
    np.fill_diagonal(conflict, False)
    conflict = conflict.astype(np.uint8)

    # --- Type 1 vectors (all mutually compatible & compatible with type 2) ---
    type1 = []
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            type1.append(v)
    type1 = np.array(type1, dtype=np.int64)           # (22, 11)

    # --- Sign generation (all 16 sign patterns for a chosen support) ---
    sign_pats = np.array(list(itertools.product((1, -1), repeat=4)), dtype=np.int64)  # (16,4)
    rng = np.random.default_rng(12345)

    # --- Simulated annealing on the support conflict graph ---
    # State: boolean array over N_SUP (selected supports).
    # Moves: 1-for-k removal, 1-for-1 add, 2-for-2 swap.
    # Objective: maximize |selected| (independent set).

    def init_state():
        # degree-ordered greedy seed
        deg = conflict.sum(axis=1)
        order = np.argsort(deg, kind='stable')
        sel = np.zeros(N_SUP, dtype=bool)
        for i in order:
            if not conflict[i, sel].any():
                sel[i] = True
        return sel

    best_sel = None
    best_size = -1

    def evaluate_state(sel):
        # returns (size, is_valid)
        size = int(sel.sum())
        # validity check: no two selected supports conflict
        idx = np.where(sel)[0]
        if len(idx) > 1:
            if conflict[np.ix_(idx, idx)].any():
                return size, False
        return size, True

    def polish(sel):
        # greedy fill: add any support with no conflict to selected
        changed = True
        while changed:
            changed = False
            cand = np.where(~sel)[0]
            if len(cand) == 0:
                break
            # supports with zero conflict to current selection
            ok = ~conflict[cand, sel].any(axis=1)
            add = cand[ok]
            if len(add) > 0:
                sel[add] = True
                changed = True
        return sel

    sel = init_state()
    sel = polish(sel)
    size, valid = evaluate_state(sel)
    if valid and size > best_size:
        best_size = size
        best_sel = sel.copy()

    # Annealing schedule
    T0, Tf = 3.0, 0.02
    n_iter = 60000
    for it in range(n_iter):
        if time.time() > deadline - 25.0:
            break
        T = T0 * (Tf / T0) ** (it / n_iter)

        # pick a move type
        r = rng.random()
        if r < 0.45:
            # 1-for-1: remove a random selected support
            idx = np.where(sel)[0]
            if len(idx) == 0:
                continue
            v = idx[rng.integers(len(idx))]
            old = sel.copy()
            sel[v] = False
            new_size = int(sel.sum())
            dE = new_size - int(old.sum())   # -1
            if dE >= 0 or rng.random() < np.exp(dE / T):
                pass
            else:
                sel = old
        elif r < 0.75:
            # 1-for-1: add a random unselected support (if it fits)
            idx = np.where(~sel)[0]
            if len(idx) == 0:
                continue
            v = idx[rng.integers(len(idx))]
            if conflict[v, sel].any():
                continue   # cannot add (would create conflict)
            sel[v] = True
        else:
            # 2-for-2 swap: remove one selected, add one unselected
            s_idx = np.where(sel)[0]
            u_idx = np.where(~sel)[0]
            if len(s_idx) == 0 or len(u_idx) == 0:
                continue
            a = s_idx[rng.integers(len(s_idx))]
            b = u_idx[rng.integers(len(u_idx))]
            # after removing a, can b be added?
            if conflict[b, sel].any() and not conflict[b, a]:
                # b conflicts with some other selected; only allowed if that is a (which we remove)
                # check: does b conflict with any selected other than a?
                if conflict[b, sel].sum() > conflict[b, a]:
                    continue
            # remove a, add b
            sel[a] = False
            sel[b] = True

        # periodic greedy polish
        if it % 2000 == 0 and it > 0:
            sel = polish(sel)
            size, valid = evaluate_state(sel)
            if valid and size > best_size:
                best_size = size
                best_sel = sel.copy()

    # Final polish + record
    sel = polish(sel)
    size, valid = evaluate_state(sel)
    if valid and size > best_size:
        best_size = size
        best_sel = sel.copy()

    # Fallback safety: ensure best_sel is valid
    if best_sel is None or not evaluate_state(best_sel)[1]:
        # rebuild a guaranteed-valid greedy set
        sel = init_state()
        sel = polish(sel)
        best_sel = sel
        best_size = int(sel.sum())

    # --- Build type-2 vectors from selected supports ---
    chosen = np.where(best_sel)[0]
    if len(chosen) > 0:
        chosen_sup = sup[chosen]                       # (m, 4)
        vecs = np.zeros((len(chosen), D), dtype=np.int64)
        for i, (row, pat) in enumerate(zip(chosen_sup, sign_pats)):
            vecs[i, row] = pat
    else:
        vecs = np.zeros((0, D), dtype=np.int64)

    # --- Combine with all type-1 vectors ---
    vectors = np.vstack([type1, vecs]) if len(vecs) else type1
    return vectors.astype(np.int64)
