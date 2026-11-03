import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    D = 11

    # ---------- Type A: one non-zero entry +/-2, norm^2 = 4 ----------
    vectors_A = []
    for i in range(D):
        v1 = [0] * D; v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D; v2[i] = -2
        vectors_A.append(v2)
    vectors_A = np.array(vectors_A, dtype=np.int64)  # (22, 11)

    # ---------- Type D: four non-zero entries +/-1, norm^2 = 4 ----------
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vectors_D.append(v)
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)  # (5280, 11)
    N = len(all_vectors_D)

    # ---------- Conflict matrix: two Type D vectors conflict if dot >= 3 ----------
    # (u-v)^2 = 8 - 2*dot >= 4  <=>  dot <= 2
    dots = all_vectors_D @ all_vectors_D.T
    conflict = (dots >= 3)
    np.fill_diagonal(conflict, True)

    # degrees
    degrees = conflict.sum(axis=1).astype(np.int32)

    best_set = None
    best_size = 0

    # ---------- Greedy helper ----------
    def greedy(order):
        sel = np.zeros(N, dtype=bool)
        cnt = 0
        for idx in order:
            if sel[idx]:
                continue
            if not np.any(conflict[idx] & sel):
                sel[idx] = True
                cnt += 1
        return sel, cnt

    # ---------- 1-swap local search ----------
    def local_search(sel, cnt):
        # Precompute: for each unselected i, count conflicts with selected
        # sel_conflict_count[i] = number of selected vectors that conflict with i
        # An unselected i can be added alone if sel_conflict_count[i] == 0
        # A swap (remove s, add i) is valid if:
        #   i conflicts with at most 1 selected vector, and that vector is s,
        #   i.e., sel_conflict_count[i] <= 1 and the conflicting selected is s.
        # We'll do a few passes.
        for _pass in range(3):
            if time.time() > deadline:
                break
            # counts of selected conflicts for each unselected
            # use matrix-vector product
            sel_conf = conflict @ sel  # (N,) int
            # unselected indices
            unselected = np.where(~sel)[0]
            if len(unselected) == 0:
                break
            # candidates that can be added without removal
            addable = unselected[sel_conf[unselected] == 0]
            if len(addable) > 0:
                # add all addable that are mutually non-conflicting (greedy)
                # since addable have 0 conflicts with selected, they only need to be
                # mutually compatible; but they may conflict with each other.
                # Just add the first few in degree order.
                addable_sorted = addable[np.argsort(degrees[addable])]
                for i in addable_sorted:
                    if time.time() > deadline:
                        break
                    if sel[i]:
                        continue
                    if not np.any(conflict[i] & sel):
                        sel[i] = True
                        cnt += 1
                # recompute sel_conf after additions
                sel_conf = conflict @ sel
            # now try swaps: for each selected s, find unselected i with
            # sel_conf[i] == 1 and the unique conflict is s, and i is "better"
            # (lower degree or just any) to improve future growth.
            # For speed, sample.
            selected_idx = np.where(sel)[0]
            if len(selected_idx) == 0:
                break
            # For each unselected i with sel_conf[i] == 1, the conflicting selected
            # is unique. We can try to swap it out.
            swap_candidates = unselected[sel_conf[unselected] == 1]
            if len(swap_candidates) > 0:
                # limit work
                if len(swap_candidates) > 400:
                    swap_candidates = swap_candidates[np.random.permutation(len(swap_candidates))[:400]]
                for i in swap_candidates:
                    if time.time() > deadline:
                        break
                    if sel[i]:
                        continue
                    # find the conflicting selected
                    conf_with_sel = np.where(conflict[i] & sel)[0]
                    if len(conf_with_sel) != 1:
                        continue
                    s = conf_with_sel[0]
                    # swap: remove s, add i. Valid because i only conflicts with s.
                    # Do it if it doesn't reduce size (it keeps size same) but may
                    # help future. Only do if degrees[i] < degrees[s] (improve).
                    if degrees[i] < degrees[s]:
                        sel[s] = False
                        sel[i] = True
                        # cnt unchanged
                        sel_conf = conflict @ sel
                        break  # recompute and continue outer pass
        return sel, cnt

    # ---------- Phase 1: greedy initializations ----------
    # min-degree order
    order = np.argsort(degrees)
    sel, cnt = greedy(order)
    sel, cnt = local_search(sel, cnt)
    if cnt > best_size:
        best_size = cnt
        best_set = sel.copy()

    # ---------- Phase 2: random restarts with local search ----------
    np.random.seed(42)
    while time.time() < deadline - 5.0:
        # weighted random order: prefer lower degree
        weights = 1.0 / (degrees + 1.0)
        weights /= weights.sum()
        order = np.random.choice(N, size=N, replace=False, p=weights)
        sel, cnt = greedy(order)
        sel, cnt = local_search(sel, cnt)
        if cnt > best_size:
            best_size = cnt
            best_set = sel.copy()

    # ---------- Combine ----------
    selected_vectors_D = all_vectors_D[best_set]
    final_vectors = np.vstack([vectors_A, selected_vectors_D])

    return final_vectors
