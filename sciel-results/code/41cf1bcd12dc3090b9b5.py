"""Kissing number in dimension 11 - vectorized MIS over equal-norm vectors."""
import numpy as np
import itertools
import time

D = 11


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # ---- Candidate pool: all vectors with norm^2 == 4 ----
    # Type A: weight-4 vectors in {-1,0,1}^11  (4 positions, +/-1)
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)
    # Type B: weight-1 vectors in {-2,0,2}^11  (1 position, +/-2)
    for pos in range(D):
        for sign in (1, -1):
            v = [0] * D
            v[pos] = 2 * sign
            v4_list.append(v)

    V = np.array(v4_list, dtype=np.int64)
    N = len(V)
    if N == 0:
        return np.zeros((0, D), dtype=np.int64)

    # ---- Conflict graph: conflict iff dot(v,w) > 2 (norm^2 = 4 -> min dist >= max norm) ----
    dots = V @ V.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    def greedy_min_degree(rng):
        """Vectorized greedy: repeatedly pick min-degree alive vertex, remove it + neighbors."""
        alive = np.ones(N, dtype=bool)
        selected = np.zeros(N, dtype=bool)
        degrees = conflicts.sum(axis=1).astype(np.int64)
        idx_all = np.arange(N)
        while True:
            if time.time() > deadline:
                break
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0:
                break
            alive_deg = degrees[alive_idx]
            min_deg = alive_deg.min()
            cands = alive_idx[alive_deg == min_deg]
            chosen = cands[rng.integers(len(cands))]
            selected[chosen] = True
            to_remove = np.where(conflicts[chosen] | (idx_all == chosen))[0]
            alive[to_remove] = False
            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                degrees[remaining] -= conflicts[remaining][:, to_remove].sum(axis=1)
        return selected

    def local_search_2swap(selected, rng, max_rounds=300):
        """Try to add one unselected vertex; if it conflicts with selected ones,
        drop those and re-add, then attempt to recover by adding a compatible
        unselected vertex. Accept if net size increases."""
        sel = selected.copy()
        sel_set = set(np.where(sel)[0].tolist())
        cur_size = len(sel_set)
        best_size = cur_size
        best_sel = sel.copy()
        unselected = np.where(~sel)[0]
        if len(unselected) == 0:
            return best_sel, best_size
        for _ in range(max_rounds):
            if time.time() > deadline - 5:
                break
            # Pick a random unselected vertex not already selected
            pool = [u for u in unselected if u not in sel_set]
            if not pool:
                break
            u = pool[rng.integers(len(pool))]
            # Neighbors of u that are currently selected
            nbrs = np.where(conflicts[u])[0]
            drop = [n for n in nbrs if n in sel_set]
            if len(drop) >= 2:
                continue  # net non-positive gain for a 1-swap
            # Remove dropped, add u
            for n in drop:
                sel_set.discard(n)
            sel_set.add(u)
            # Try to recover: add any unselected vertex compatible with all in sel_set
            improved = False
            for w in unselected:
                if w in sel_set:
                    continue
                # check w conflicts with no member of sel_set
                if not np.any(conflicts[w][list(sel_set)]):
                    sel_set.add(w)
                    improved = True
                    break
            new_size = len(sel_set)
            if new_size > best_size:
                best_size = new_size
                best_sel = np.zeros(N, dtype=bool)
                best_sel[list(sel_set)] = True
            else:
                # revert
                sel_set.discard(u)
                for n in drop:
                    sel_set.add(n)
        return best_sel, best_size

    # ---- Multiple random restarts of greedy, keep best, then local search ----
    best_selected = None
    best_count = 0
    n_restarts = 40
    for r in range(n_restarts):
        if time.time() > deadline - 15:
            break
        rng = np.random.default_rng(12345 + r * 97)
        sel = greedy_min_degree(rng)
        cnt = int(sel.sum())
        if cnt > best_count:
            best_count = cnt
            best_selected = sel.copy()

    if best_selected is None:
        best_selected = greedy_min_degree(np.random.default_rng(0))
        best_count = int(best_selected.sum())

    # ---- Local search refinement on the best solution ----
    if time.time() < deadline - 10:
        rng = np.random.default_rng(999)
        best_selected, best_count = local_search_2swap(best_selected, rng, max_rounds=600)

    result = V[best_selected]
    return result
