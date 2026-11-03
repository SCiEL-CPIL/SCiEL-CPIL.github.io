import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    D = 11

    # Type A: Vectors with one non-zero entry +/- 2 (norm^2 = 4)
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)

    # Type D: Vectors with four non-zero entries +/- 1 (norm^2 = 4)
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_sup = len(supports)  # 330
    N_D = N_sup * 16  # 5280

    # Build all Type D vectors
    all_D = np.zeros((N_D, D), dtype=np.int8)
    for si, combo in enumerate(supports):
        base = si * 16
        for s in range(16):
            v = [0] * D
            for j, idx in enumerate(combo):
                v[idx] = 1 if (s >> j) & 1 else -1
            all_D[base + s] = v

    # Compute conflict matrix: conflict if dot > 2 (i.e., dot >= 3)
    # dot = sum of products over shared positions
    dots = all_D.astype(np.int16) @ all_D.astype(np.int16).T
    conflict = (dots > 2).astype(np.int8)
    np.fill_diagonal(conflict, 0)

    # Precompute: for each support, which other supports have overlap >= 3
    # (These are the only supports that can conflict with a given support)
    # overlap[i,j] = number of common indices between support i and support j
    support_arr = np.array(supports, dtype=np.int8)  # (330, 4)
    # Build indicator matrix: sup_ind[i, x] = 1 if x in support i
    sup_ind = np.zeros((N_sup, D), dtype=np.int8)
    for i in range(N_sup):
        for x in supports[i]:
            sup_ind[i, x] = 1
    overlaps = sup_ind @ sup_ind.T  # (330, 330)

    # For each support, list of other supports with overlap >= 3
    # (conflicts can only happen between these)
    conflict_support_mask = (overlaps >= 3)
    conflict_support_mask.fill_diagonal(False)

    # Now do local search on the full 5280-vector conflict graph
    # State: boolean array of length N_D indicating selected vectors
    best_set = None
    best_count = 0

    N = N_D

    def count_selected(sel):
        return np.sum(sel)

    def valid_set(sel):
        # Check no two selected vectors conflict
        idx = np.where(sel)[0]
        if len(idx) <= 1:
            return True
        sub_conflict = conflict[np.ix_(idx, idx)]
        return not np.any(sub_conflict)

    # Greedy initialization: pick random vectors, skip conflicts
    def greedy_init(rng):
        sel = np.zeros(N, dtype=bool)
        available = np.arange(N)
        while len(available) > 0:
            if rng.random() < 0.5:
                # Pick random from available
                pos = rng.integers(0, len(available))
            else:
                pos = 0
            v = available[pos]
            sel[v] = True
            # Remove v and all its neighbors from available
            neighbors = np.where(conflict[v] == 1)[0]
            to_remove = np.concatenate([[v], neighbors])
            available = np.setdiff1d(available, to_remove, assume_unique=False)
        return sel

    # Local search: 1-swap moves (remove one, add one)
    def local_search(sel, rng, max_iters, time_limit):
        nonlocal best_set, best_count
        cur_count = np.sum(sel)
        if cur_count > best_count:
            best_count = cur_count
            best_set = sel.copy()

        for it in range(max_iters):
            if time.time() > time_limit:
                break

            # Find all vectors that could be added (not selected, no conflict with selected)
            selected_idx = np.where(sel)[0]
            if len(selected_idx) == 0:
                # Add any random vector
                v = rng.integers(0, N)
                sel[v] = True
                continue

            # For each non-selected vector, check if it conflicts with any selected
            # Vectorized: conflict with selected = any(conflict[non_sel, selected])
            non_sel_mask = ~sel
            non_sel_idx = np.where(non_sel_mask)[0]
            if len(non_sel_idx) == 0:
                break

            # Check which non-selected vectors are compatible with all selected
            # compatible[v] = not any(conflict[v, s] for s in selected)
            compat = np.ones(len(non_sel_idx), dtype=bool)
            # Batch check to avoid O(N^2)
            # For each selected, mark its non-selected neighbors as incompatible
            for s in selected_idx:
                neigh = np.where(conflict[s] == 1)[0]
                neigh_non_sel = neigh[non_sel_mask]
                if len(neigh_non_sel) > 0:
                    positions = np.searchsorted(non_sel_idx, neigh_non_sel)
                    compat[positions] = False

            addable = non_sel_idx[compat]
            if len(addable) == 0:
                # Try removing a random selected and adding something
                # Or just try random swaps
                if cur_count == 0:
                    break
                # Remove a random selected
                rem_pos = rng.integers(0, cur_count)
                rem_v = selected_idx[rem_pos]
                sel[rem_v] = False
                cur_count -= 1
                if cur_count > best_count:
                    best_count = cur_count
                    best_set = sel.copy()
                continue

            # Try adding a random addable vector
            add_v = addable[rng.integers(0, len(addable))]
            sel[add_v] = True
            cur_count += 1
            if cur_count > best_count:
                best_count = cur_count
                best_set = sel.copy()

        return sel

    # Run multiple restarts
    rng = np.random.default_rng(42)
    num_restarts = 50
    iters_per_restart = 5000

    for r in range(num_restarts):
        if time.time() > deadline:
            break

        # Different initialization strategies
        if r < 10:
            # Pure random greedy
            sel = greedy_init(rng)
        elif r < 20:
            # Start with a few random, then greedy
            sel = np.zeros(N, dtype=bool)
            n_seed = rng.integers(1, 20)
            seeds = rng.choice(N, size=n_seed, replace=False)
            for s in seeds:
                if not np.any(conflict[s][sel]):
                    sel[s] = True
            # Then greedy fill
            available = np.where(~sel)[0]
            while len(available) > 0:
                pos = rng.integers(0, len(available))
                v = available[pos]
                if not np.any(conflict[v][sel]):
                    sel[v] = True
                    neighbors = np.where(conflict[v] == 1)[0]
                    to_remove = np.concatenate([[v], neighbors])
                    available = np.setdiff1d(available, to_remove, assume_unique=False)
                else:
                    available = np.delete(available, pos)
        else:
            # Random start
            sel = np.zeros(N, dtype=bool)
            n_seed = rng.integers(5, 50)
            seeds = rng.choice(N, size=n_seed, replace=False)
            for s in seeds:
                if not np.any(conflict[s][sel]):
                    sel[s] = True

        # Local search
        sel = local_search(sel, rng, iters_per_restart, deadline)

    # Combine Type A (all 22) with best Type D set
    vectors_A_arr = np.array(vectors_A, dtype=np.int64)
    if best_set is not None:
        d_selected = all_D[best_set].astype(np.int64)
    else:
        d_selected = np.empty((0, D), dtype=np.int64)

    final_vectors = np.vstack([vectors_A_arr, d_selected])

    return final_vectors
