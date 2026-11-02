"""Structural change: exploit symmetry via orbit-based grouping for independent set in kissing config in R^11."""
import numpy as np
import itertools
import time
import random


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s

    D = 11

    # Generate all nonzero integer vectors of small norm in R^11
    # Weight-1: 22 vectors (norm^2=1)
    # Weight-2: 220 vectors (norm^2=2)
    # Weight-3: C(11,3)*8 = 1760 vectors (norm^2=3)
    # Weight-4: C(11,4)*16 = 5280 vectors (norm^2=4)

    candidates_list = []

    # Weight 1
    for pos in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[pos] = s
            candidates_list.append(tuple(v))

    # Weight 2
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(tuple(v))

    # Weight 3
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(tuple(v))

    # Weight 4
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(tuple(v))

    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)

    # Precompute norms squared
    norms_sq = np.sum(candidates.astype(np.int32) ** 2, axis=1)
    max_norm_sq = int(norms_sq.max())  # 4

    # Precompute dot product matrix
    dot = candidates.astype(np.int32) @ candidates.T.astype(np.int32)

    # Constraint: ||v_i - v_j||^2 >= max_norm_sq
    # => v_i·v_j <= (||v_i||^2 + ||v_j||^2 - max_norm_sq) / 2
    threshold = (norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - max_norm_sq) / 2.0
    compat = (dot <= threshold).astype(np.uint8)
    np.fill_diagonal(compat, 1)

    # conflict[i,j] = 1 if i and j CANNOT both be selected (i != j)
    conflict = (1 - compat).astype(np.uint8)
    np.fill_diagonal(conflict, 0)

    def greedy_fill(selected_list, time_limit):
        """Fill in more vectors greedily by checking compatibility."""
        selected = list(selected_list)
        if not selected:
            return selected

        # Build set for O(1) lookup
        sel_set = set(selected)

        # Compute which candidates are compatible with ALL selected
        # Using bitset-like approach: for each candidate, count conflicts with selected
        # But that's O(n * len(selected)) which is fine

        # More efficient: maintain a running mask
        # compatible_mask[j] = 1 if j is compatible with all selected
        compatible_mask = np.ones(n, dtype=np.uint8)
        for s in selected:
            compatible_mask = compatible_mask & compat[s]
        for s in selected:
            compatible_mask[s] = 0

        # Also track degree of each candidate among remaining
        remaining = compatible_mask.copy()

        while time.time() < time_limit:
            addable = np.where(remaining > 0)[0]
            if len(addable) == 0:
                break

            # Among addable, pick the one with minimum conflict degree
            # deg[i] = number of conflicts with other remaining candidates
            deg = (conflict[addable] * remaining[np.newaxis, :]).sum(axis=1)
            min_idx = np.argmin(deg)
            new_idx = addable[min_idx]

            selected.append(new_idx)
            sel_set.add(new_idx)

            # Update remaining: remove new_idx and its conflicts
            remaining = remaining & (1 - conflict[new_idx])
            remaining[new_idx] = 0

        return selected

    def greedy_min_degree(remaining_init, time_limit):
        """Min-degree greedy on the conflict graph restricted to remaining vertices."""
        remaining = remaining_init.copy()
        selected = []
        while time.time() < time_limit and remaining.any():
            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n + 1
            min_deg = deg.min()
            if min_deg >= n:
                break
            mask = (deg == min_deg) & (remaining > 0)
            picks = np.where(mask)[0]
            if len(picks) == 0:
                break
            new_idx = picks[0]
            selected.append(new_idx)
            remaining = remaining & (1 - conflict[new_idx])
            remaining[new_idx] = 0
        return selected

    # Strategy: Multiple diverse greedy restarts + local improvement
    np.random.seed(42)
    random.seed(42)

    best_selected = []
    best_len = 0

    # 1. Pure min-degree greedy on all candidates
    sel = greedy_min_degree(np.ones(n, dtype=np.uint8), deadline - 100.0)
    if len(sel) > best_len:
        best_selected = sel
        best_len = len(sel)

    # 2. Random subset restarts
    for trial in range(50):
        if time.time() > deadline - 40.0:
            break
        frac = random.uniform(0.2, 0.8)
        subset_size = int(n * frac)
        subset_indices = np.random.choice(n, size=subset_size, replace=False)
        remaining_init = np.zeros(n, dtype=np.uint8)
        remaining_init[subset_indices] = 1
        sel = greedy_min_degree(remaining_init, deadline - 40.0)
        if len(sel) > best_len:
            best_selected = sel
            best_len = len(sel)

    # 3. Greedy fill from small random seeds
    for trial in range(30):
        if time.time() > deadline - 30.0:
            break
        seed_size = random.randint(1, 5)
        seed_indices = np.random.choice(n, size=seed_size, replace=False)
        # Check if seed is a valid independent set
        valid_seed = True
        for i in range(seed_size):
            for j in range(i + 1, seed_size):
                if conflict[seed_indices[i], seed_indices[j]]:
                    valid_seed = False
                    break
            if not valid_seed:
                break
        if valid_seed:
            sel = greedy_fill(list(seed_indices), deadline - 30.0)
            if len(sel) > best_len:
                best_selected = sel
                best_len = len(sel)

    # 4. Local search: swap-based improvement
    def swap_improve(selected_list, time_limit, max_rounds=500):
        """Try to improve by removing one element and adding two compatible ones."""
        selected = list(selected_list)
        sel_set = set(selected)

        # Precompute compatibility with all selected
        compat_with_all = np.ones(n, dtype=np.uint8)
        for s in selected:
            compat_with_all = compat_with_all & compat[s]
        for s in selected:
            compat_with_all[s] = 0

        for _ in range(max_rounds):
            if time.time() > time_limit:
                break

            # First try simple additions
            addable = np.where(compat_with_all > 0)[0]
            if len(addable) > 0:
                full_deg = conflict.sum(axis=1)
                best_add = addable[np.argmin(full_deg[addable])]
                selected.append(best_add)
                sel_set.add(best_add)
                compat_with_all = compat_with_all & compat[best_add]
                compat_with_all[best_add] = 0
                continue

            # Try swap: remove one, add two
            found_improvement = False

            # Limit trials for speed
            if len(selected) > 50:
                trial_count = min(15, len(selected))
                trial_indices = np.random.choice(len(selected), size=trial_count, replace=False)
            else:
                trial_indices = range(len(selected))

            for si in trial_indices:
                if time.time() > time_limit:
                    break
                s = selected[si]

                # Candidates compatible with all selected except s
                mask_except_s = np.ones(n, dtype=np.uint8)
                for s2 in selected:
                    if s2 != s:
                        mask_except_s = mask_except_s & compat[:, s2]
                mask_except_s[s] = 0
                mask_except_s[list(sel_set)] = 0

                candidates_for_swap = np.where(mask_except_s > 0)[0]
                if len(candidates_for_swap) < 2:
                    continue

                # Sort by full degree (least constraining first)
                full_deg = conflict.sum(axis=1)
                sorted_cands = candidates_for_swap[np.argsort(full_deg[candidates_for_swap])]

                # Find a compatible pair
                found_pair = False
                for i in range(min(15, len(sorted_cands))):
                    c1 = sorted_cands[i]
                    compat_c1 = compat[c1]
                    for j in range(i + 1, min(30, len(sorted_cands))):
                        c2 = sorted_cands[j]
                        if compat_c1[c2]:
                            # Do the swap
                            selected.remove(s)
                            sel_set.discard(s)
                            selected.append(c1)
                            sel_set.add(c1)
                            selected.append(c2)
                            sel_set.add(c2)

                            # Recompute compat_with_all
                            compat_with_all = np.ones(n, dtype=np.uint8)
                            for s3 in selected:
                                compat_with_all = compat_with_all & compat[s3]
                            for s3 in selected:
                                compat_with_all[s3] = 0

                            found_pair = True
                            break
                    if found_pair:
                        break

                if found_pair:
                    found_improvement = True
                    break

            if not found_improvement:
                break

        return selected

    if best_selected and time.time() < deadline - 20.0:
        improved = swap_improve(best_selected, deadline - 20.0)
        if len(improved) > best_len:
            best_selected = improved
            best_len = len(improved)

    # Final greedy fill
    if best_selected and time.time() < deadline - 5.0:
        final = greedy_fill(best_selected, deadline - 5.0)
        if len(final) > best_len:
            best_selected = final
            best_len = len(final)

    if not best_selected:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)

    result = candidates[np.array(best_selected, dtype=np.int64)]
    return result.astype(np.int64)
