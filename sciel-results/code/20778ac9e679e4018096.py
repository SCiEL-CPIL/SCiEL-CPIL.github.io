"""Combined program: Multi-start greedy with local search for kissing number in dimension 11.
Uses expanded candidate set (norms 1,2,4) and efficient local search."""
import numpy as np
import itertools
import time


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    D = 11

    # Generate all nonzero integer vectors with squared norm <= 4
    candidates_list = []

    # Type 1: One 1, rest 0 (squared norm = 1)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = s
            candidates_list.append(v)

    # Type 2: Two 1s, rest 0 (squared norm = 2)
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Type 3: Four 1s, rest 0 (squared norm = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Type 4: One 2, rest 0 (squared norm = 4)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates_list.append(v)

    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)

    # Precompute norms squared
    norms_sq = np.sum(candidates.astype(np.int32) ** 2, axis=1)
    max_norm_sq = int(norms_sq.max())  # 4

    # Precompute dot product matrix
    dot = candidates.astype(np.int32) @ candidates.T.astype(np.int32)

    # Conflict matrix: 1 if i and j CANNOT both be selected
    # Constraint: ||v_i - v_j||^2 >= max_norm_sq
    # => v_i·v_j <= (||v_i||^2 + ||v_j||^2 - max_norm_sq) / 2
    threshold = (norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - max_norm_sq) / 2.0
    compat = (dot <= threshold).astype(np.uint8)
    np.fill_diagonal(compat, 1)

    # conflict[i,j] = 1 if i and j CANNOT both be selected
    conflict = (1 - compat).astype(np.uint8)
    np.fill_diagonal(conflict, 0)

    # Precompute degree (number of conflicts) for each vertex
    full_deg = conflict.sum(axis=1).astype(np.int32)

    def greedy_randomized(seed):
        """Randomized min-degree greedy."""
        rng = np.random.default_rng(seed)
        remaining = np.ones(n, dtype=bool)
        deg = full_deg.copy()
        selected = []

        while remaining.any():
            if time.time() > deadline:
                break

            active_indices = np.where(remaining)[0]
            if len(active_indices) == 0:
                break

            active_deg = deg[active_indices]
            min_deg = np.min(active_deg)
            candidates_for_pick = active_indices[active_deg == min_deg]
            new_idx = rng.choice(candidates_for_pick)

            selected.append(new_idx)

            # Remove new_idx and its conflicts from remaining
            neighbors = np.where(conflict[new_idx] > 0)[0]
            to_remove = np.concatenate([[new_idx], neighbors[remaining[neighbors]]])
            remaining[to_remove] = False

            # Update degrees
            for rem in to_remove:
                if remaining[rem] == False:
                    continue
                deg[remaining] -= conflict[rem, remaining]

        return selected

    def local_search(selected_list, time_limit):
        """Local search: 1-swap and 2-swap."""
        selected = list(selected_list)
        sel_set = set(selected)

        # Precompute compat_with_all: 1 if j is compatible with all currently selected
        compat_with_all = np.ones(n, dtype=np.uint8)
        for s in selected:
            compat_with_all = compat_with_all & compat[s]
        for s in selected:
            compat_with_all[s] = 0

        improved = True
        max_iters = 100
        iters = 0

        while improved and iters < max_iters and time.time() < time_limit:
            improved = False
            iters += 1

            # 1. Try to add a vertex (if any)
            addable = np.where(compat_with_all > 0)[0]
            if len(addable) > 0:
                # Pick the one with min full degree (least constraining)
                best_add = addable[np.argmin(full_deg[addable])]
                selected.append(best_add)
                sel_set.add(best_add)
                compat_with_all = compat_with_all & compat[best_add]
                compat_with_all[best_add] = 0
                improved = True
                continue

            # 2. Try 1-swap: Remove one, add one
            # For each selected s, check if there is a candidate j compatible with (selected - {s})
            # and NOT compatible with s.

            # To speed up, only try swaps for a subset of s if selected is large
            if len(selected) > 50:
                trial_indices = np.random.choice(len(selected), size=min(20, len(selected)), replace=False)
            else:
                trial_indices = range(len(selected))

            for si in trial_indices:
                if time.time() > time_limit:
                    break
                s = selected[si]

                # Compute mask of candidates compatible with all selected except s
                mask_except_s = np.ones(n, dtype=np.uint8)
                for s2 in selected:
                    if s2 != s:
                        mask_except_s = mask_except_s & compat[:, s2]

                # We want j such that:
                # 1. j is in mask_except_s
                # 2. j is NOT in selected
                # 3. j is incompatible with s (otherwise it would be in compat_with_all)

                mask_except_s[list(sel_set)] = 0  # Exclude selected
                candidates_for_swap = np.where(mask_except_s > 0)[0]

                if len(candidates_for_swap) > 0:
                    # Pick the best candidate (min full degree)
                    best_swap = candidates_for_swap[np.argmin(full_deg[candidates_for_swap])]

                    # Perform swap
                    selected.remove(s)
                    sel_set.discard(s)
                    selected.append(best_swap)
                    sel_set.add(best_swap)

                    # Recompute compat_with_all
                    compat_with_all = np.ones(n, dtype=np.uint8)
                    for s3 in selected:
                        compat_with_all = compat_with_all & compat[s3]
                    for s3 in selected:
                        compat_with_all[s3] = 0

                    improved = True
                    break  # Restart loop with new state

            # 3. Try 2-swap: Remove one, add two
            if not improved and time.time() < time_limit:
                # Similar to 1-swap, but we need two compatible candidates from mask_except_s
                if len(selected) > 50:
                    trial_indices = np.random.choice(len(selected), size=min(10, len(selected)), replace=False)
                else:
                    trial_indices = range(len(selected))

                for si in trial_indices:
                    if time.time() > time_limit:
                        break
                    s = selected[si]

                    mask_except_s = np.ones(n, dtype=np.uint8)
                    for s2 in selected:
                        if s2 != s:
                            mask_except_s = mask_except_s & compat[:, s2]

                    mask_except_s[list(sel_set)] = 0
                    candidates_for_swap = np.where(mask_except_s > 0)[0]

                    if len(candidates_for_swap) >= 2:
                        # Find a pair (c1, c2) in candidates_for_swap that are compatible
                        # Sort by full degree
                        sorted_cands = candidates_for_swap[np.argsort(full_deg[candidates_for_swap])]

                        found_pair = False
                        # Check first few candidates
                        for i in range(min(10, len(sorted_cands))):
                            c1 = sorted_cands[i]
                            # Find c2 compatible with c1
                            compat_with_c1 = compat[c1]
                            # We want c2 in sorted_cands such that compat[c1, c2] == 1
                            for j in range(i + 1, min(i + 20, len(sorted_cands))):
                                c2 = sorted_cands[j]
                                if compat_with_c1[c2]:
                                    # Do the swap
                                    selected.remove(s)
                                    sel_set.discard(s)
                                    selected.append(c1)
                                    sel_set.add(c1)
                                    selected.append(c2)
                                    sel_set.add(c2)

                                    compat_with_all = np.ones(n, dtype=np.uint8)
                                    for s3 in selected:
                                        compat_with_all = compat_with_all & compat[s3]
                                    for s3 in selected:
                                        compat_with_all[s3] = 0

                                    improved = True
                                    found_pair = True
                                    break
                            if found_pair:
                                break
                    if improved:
                        break

        return selected

    best_selected = []
    best_size = 0

    # Multi-start
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break

        # Greedy
        sel = greedy_randomized(seed=r * 1000)

        # Local Search
        sel = local_search(sel, deadline - 2.0)

        if len(sel) > best_size:
            best_size = len(sel)
            best_selected = sel

    if not best_selected:
        # Fallback: return a single vector
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

    result = candidates[best_selected]
    return result.astype(np.int64)
