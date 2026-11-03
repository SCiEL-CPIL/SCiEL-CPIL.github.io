import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    D = 11

    # Generate ALL nonzero integer vectors with squared norm <= 4
    candidates = []

    # Type A: one coordinate is +/-1 (norm^2 = 1)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            candidates.append(v)

    # Type B: two coordinates are +/-1 (norm^2 = 2)
    for idxs in itertools.combinations(range(D), 2):
        for signs in itertools.product((1, -1), repeat=2):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)

    # Type C: one coordinate is +/-2 (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)

    # Type D: four coordinates are +/-1 (norm^2 = 4)
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)

    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)

    # Squared norms per vector
    norms_sq = np.sum(candidates.astype(np.int64) ** 2, axis=1)

    # Dot product matrix
    M = candidates.astype(np.int64) @ candidates.astype(np.int64).T

    # Conflict: pair (i,j) incompatible if dist^2 < max_norm^2 = 4
    # dist^2 = n_i + n_j - 2*M[i,j]  < 4  =>  2*M[i,j] > n_i + n_j - 4
    norms_sum = norms_sq[:, None] + norms_sq[None, :]
    Conflict = (2 * M > norms_sum - 4).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)

    # Precompute neighbor lists (int arrays) for fast access
    neighbors = [np.where(Conflict[i] == 1)[0] for i in range(N)]

    def run_greedy(seed):
        if time.time() > deadline:
            return []
        rng = np.random.default_rng(seed)
        active = np.ones(N, dtype=bool)
        deg = Conflict.sum(axis=1).astype(np.int64)
        selected = []

        while active.any():
            if time.time() > deadline:
                break
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            active_deg = deg[active_indices]
            min_deg = np.min(active_deg)
            candidates_for_pick = active_indices[active_deg == min_deg]
            idx = int(rng.choice(candidates_for_pick))
            selected.append(idx)

            removed_mask = np.zeros(N, dtype=bool)
            removed_mask[idx] = True
            removed_mask[neighbors[idx]] = True
            removed_mask &= active | removed_mask  # only remove active ones
            removed_mask[idx] = True

            removed_indices = np.where(removed_mask)[0]
            if len(removed_indices) > 0:
                removal_counts = Conflict[removed_indices, :].sum(axis=0).astype(np.int64)
                deg[active] -= removal_counts[active]
            active[removed_mask] = False

        return selected

    def local_search_fast(selected_indices):
        if time.time() > deadline:
            return selected_indices
        selected_set = set(selected_indices)
        # membership array for O(1) checks
        in_set = np.zeros(N, dtype=bool)
        for x in selected_set:
            in_set[x] = True

        improved = True
        max_rounds = 60
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline:
            improved = False
            rounds += 1
            cur = np.array(list(selected_set), dtype=np.int64)
            rng_ls = np.random.default_rng()
            order = rng_ls.permutation(len(cur))
            cur = cur[order]

            for i_idx in range(len(cur)):
                if time.time() > deadline:
                    return list(selected_set)
                i = int(cur[i_idx])
                # neighbors of i not currently in the set
                nbrs = neighbors[i]
                valid_js = nbrs[~in_set[nbrs]]
                if len(valid_js) < 2:
                    continue
                # compatible with all of selected_set \ {i}
                rest = cur[cur != i]
                if len(rest) > 0:
                    conflict_with_rest = Conflict[valid_js, rest].any(axis=1)
                    compatible = valid_js[~conflict_with_rest]
                else:
                    compatible = valid_js
                if len(compatible) < 2:
                    continue
                # find a mutually compatible pair
                found = False
                for p in range(len(compatible)):
                    if found:
                        break
                    j1 = int(compatible[p])
                    rest_comp = compatible[p + 1:]
                    if len(rest_comp) == 0:
                        continue
                    comp_with_j1 = Conflict[j1, rest_comp] == 0
                    if comp_with_j1.any():
                        j2 = int(rest_comp[np.where(comp_with_j1)[0][0]])
                        # swap i -> j1, j2
                        selected_set.remove(i)
                        in_set[i] = False
                        selected_set.add(j1)
                        in_set[j1] = True
                        selected_set.add(j2)
                        in_set[j2] = True
                        improved = True
                        found = True
                        break
                if found:
                    break
        return list(selected_set)

    best_set = []
    best_size = 0

    # Phase 1: many greedy restarts + local search
    s = 0
    while time.time() < deadline - 5:
        res = run_greedy(s * 1000 + 17)
        if not res:
            s += 1
            continue
        res = local_search_fast(res)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res
        s += 1

    # Phase 2: extra local-search passes on the best set
    if best_set:
        for _ in range(5):
            if time.time() > deadline - 1:
                break
            improved = local_search_fast(best_set)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved

    if not best_set:
        best_set = [0]

    return candidates[best_set].astype(np.int64)
