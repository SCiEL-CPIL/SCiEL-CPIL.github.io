"""Kissing number D=11: maximize vectors with min pairwise dist >= max norm."""
import time
import numpy as np
from itertools import combinations, product

D = 11

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 105.0

    # Generate W4 vectors: C(11,4)*16 = 5280
    w4_list = []
    for quad in combinations(range(D), 4):
        for signs in product((1, -1), repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(quad, signs):
                v[pos] = s
            w4_list.append(v)
    w4_arr = np.array(w4_list)
    n = len(w4_arr)

    # W4-W4: compatible iff dot <= 0
    dots = w4_arr @ w4_arr.T
    conflict = dots > 0
    np.fill_diagonal(conflict, False)
    del dots

    # Precompute conflict neighbor lists
    conflict_sets = [np.where(conflict[i])[0] for i in range(n)]
    deg = np.array([len(s) for s in conflict_sets], dtype=np.int32)

    # Greedy min-degree independent set
    selected = np.zeros(n, dtype=bool)
    remaining = np.ones(n, dtype=bool)
    order = np.argsort(deg)

    for idx in order:
        if time.time() > DEADLINE:
            break
        if not remaining[idx]:
            continue
        if not selected[conflict_sets[idx]].any():
            selected[idx] = True
            cset = conflict_sets[idx]
            remaining[cset] = False
            deg[remaining] -= conflict[idx, remaining].sum(axis=1)

    k = selected.sum()
    print(f"Greedy: {k}. Time: {time.time()-t0:.1f}s", flush=True)

    # Fill: add any compatible unselected vertex
    for idx in range(n):
        if not selected[idx] and not selected[conflict_sets[idx]].any():
            selected[idx] = True
    k = selected.sum()
    print(f"Fill: {k}. Time: {time.time()-t0:.1f}s", flush=True)

    # Local search: remove one selected, add all newly compatible
    sel_list = list(np.where(selected)[0])
    for it in range(500):
        if time.time() > DEADLINE:
            break
        improved = False
        n_try = min(len(sel_list), 200)
        if n_try == 0:
            break
        trial = np.random.choice(len(sel_list), n_try, replace=False)
        
        for ti in trial:
            ri = sel_list[ti]
            cset = conflict_sets[ri]
            candidates = cset[~selected[cset]]
            if len(candidates) == 0:
                continue
            sel_mask = selected.copy()
            sel_mask[ri] = False
            cand_conflicts = conflict[candidates, sel_mask].any(axis=1)
            addable = candidates[~cand_conflicts]
            gain = len(addable) - 1
            if gain > 0:
                selected[ri] = False
                for j in addable:
                    selected[j] = True
                sel_list = list(np.where(selected)[0])
                improved = True
                if it % 100 == 0:
                    print(f"Iter {it}: k={len(sel_list)}. Time: {time.time()-t0:.1f}s", flush=True)
        
        if not improved:
            break

    k = selected.sum()
    print(f"Final: {k}. Time: {time.time()-t0:.1f}s", flush=True)
    return w4_arr[selected]

# EVOLVE-BLOCK-END
