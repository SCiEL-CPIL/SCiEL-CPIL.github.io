"""Structural change: geometric constructive strategy using the D11 root lattice
plus a second layer of norm-2 vectors, with efficient vectorized local search.

The D11 lattice (all integer vectors of even weight-2, i.e. exactly two ±1 entries
with opposite signs, plus vectors with one ±2) gives a dense, algebraically
guaranteed kissing configuration. We then attempt to add additional norm-2 vectors
(four ±1 entries) that are compatible with the existing set, using vectorized
conflict checks and a min-degree greedy fill."""
import numpy as np
import itertools
import time


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s

    D = 11

    # ---- Generate candidate vectors ----
    # Layer 1: D11-like roots
    #   Type A: vectors with exactly one entry ±2 (norm^2 = 4)
    #   Type B: vectors with exactly two entries of opposite sign ±1 (norm^2 = 2)
    # Layer 2: additional norm-2 vectors with four ±1 entries (norm^2 = 4)

    candidates_list = []

    # Type A: one ±2
    for pos in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[pos] = 2 * s
            candidates_list.append(tuple(v))

    # Type B: two opposite-sign ±1
    for pos1 in range(D):
        for pos2 in range(pos1 + 1, D):
            # (pos1, pos2) = (+1, -1)
            v1 = [0] * D
            v1[pos1] = 1
            v1[pos2] = -1
            candidates_list.append(tuple(v1))
            # (pos1, pos2) = (-1, +1)
            v2 = [0] * D
            v2[pos1] = -1
            v2[pos2] = 1
            candidates_list.append(tuple(v2))

    # Type C: four ±1 entries (all sign combinations)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(tuple(v))

    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)

    # Precompute squared norms
    norms_sq = np.sum(candidates.astype(np.int32) ** 2, axis=1)
    max_norm_sq = int(norms_sq.max())  # should be 4

    # Precompute dot product matrix
    dot = candidates.astype(np.int32) @ candidates.T.astype(np.int32)

    # Constraint: ||v_i - v_j||^2 >= max_norm_sq
    # => ||v_i||^2 + ||v_j||^2 - 2*v_i·v_j >= max_norm_sq
    # => v_i·v_j <= (||v_i||^2 + ||v_j||^2 - max_norm_sq) / 2
    threshold = (norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - max_norm_sq) / 2.0
    compat = (dot <= threshold).astype(np.uint8)
    np.fill_diagonal(compat, 1)

    # conflict[i,j] = 1 if i and j CANNOT both be selected
    conflict = (1 - compat).astype(np.uint8)
    np.fill_diagonal(conflict, 0)

    # ---- Phase 1: Start with all Type A + Type B vectors (the D11 root system) ----
    # These are the first (2*D + 2*C(D,2)) candidates
    n_type_a = 2 * D
    n_type_b = 2 * itertools.combinations(range(D), 2).__len__()  # 2 * C(11,2) = 110
    root_count = n_type_a + n_type_b  # 22 + 110 = 132

    # Verify that all root vectors are mutually compatible
    root_indices = list(range(root_count))
    # Check: for all pairs in root set, conflict should be 0
    # (They should be, by construction of D11 roots, but let's verify and filter if needed)
    root_conflict = conflict[np.ix_(root_indices, root_indices)]
    if root_conflict.sum() > 0:
        # Some roots conflict; fall back to greedy on all candidates
        root_indices = []
        for i in range(root_count):
            if conflict[i, :root_count].sum() == 0 or i not in root_indices:
                # Check compatibility with already-selected roots
                if all(conflict[i, r] == 0 for r in root_indices):
                    root_indices.append(i)

    selected = list(root_indices)
    if not selected:
        selected = []

    # Build compatibility mask: which candidates are compatible with ALL selected
    if selected:
        compat_mask = np.ones(n, dtype=np.uint8)
        for s in selected:
            compat_mask = compat_mask & compat[s]
        for s in selected:
            compat_mask[s] = 0
    else:
        compat_mask = np.ones(n, dtype=np.uint8)

    # ---- Phase 2: Greedy fill with min-degree heuristic ----
    # Work only on candidates that are compatible with all selected
    remaining = compat_mask.copy()

    # Precompute full degree for sorting
    full_deg = conflict.sum(axis=1).astype(np.int64)

    while time.time() < deadline - 10.0:
        addable = np.where(remaining > 0)[0]
        if len(addable) == 0:
            break

        # Among addable, pick the one with minimum conflict degree (among remaining)
        deg = (conflict[addable] * remaining[np.newaxis, :]).sum(axis=1).astype(np.int64)
        min_idx = np.argmin(deg)
        new_idx = addable[min_idx]

        selected.append(new_idx)

        # Update remaining: remove new_idx and its conflicts
        remaining = remaining & (1 - conflict[new_idx])
        remaining[new_idx] = 0

    # ---- Phase 3: Local search via swap improvement ----
    # Try to remove one selected element and add two compatible ones
    sel_set = set(selected)
    improved = True
    max_rounds = 200

    for _round in range(max_rounds):
        if time.time() > deadline - 5.0:
            break
        improved = False

        # Recompute compat_with_all
        if selected:
            compat_with_all = np.ones(n, dtype=np.uint8)
            for s in selected:
                compat_with_all = compat_with_all & compat[s]
            for s in selected:
                compat_with_all[s] = 0
        else:
            compat_with_all = np.ones(n, dtype=np.uint8)

        # First, try simple additions (shouldn't find any if greedy exhausted, but check)
        addable = np.where(compat_with_all > 0)[0]
        if len(addable) > 0:
            best_add = addable[np.argmin(full_deg[addable])]
            selected.append(best_add)
            sel_set.add(best_add)
            improved = True
            continue

        # Try swap: for each selected element (sampled), try to replace with two
        if len(selected) > 30:
            np.random.seed(_round)
            trial_count = min(20, len(selected))
            trial_indices = np.random.choice(len(selected), size=trial_count, replace=False)
        else:
            trial_indices = range(len(selected))

        for si in trial_indices:
            if time.time() > deadline - 5.0:
                break
            s = selected[si]

            # Candidates compatible with all selected except s
            # mask_except_s[j] = 1 if j is compatible with all selected \ {s}
            mask_except_s = np.ones(n, dtype=np.uint8)
            for s2 in selected:
                if s2 != s:
                    mask_except_s = mask_except_s & compat[s2]
            mask_except_s[s] = 0
            # Exclude already-selected
            for s3 in selected:
                mask_except_s[s3] = 0

            candidates_for_swap = np.where(mask_except_s > 0)[0]
            if len(candidates_for_swap) < 2:
                continue

            # Sort by full degree (least constraining first)
            sorted_cands = candidates_for_swap[np.argsort(full_deg[candidates_for_swap])]
            sorted_cands = sorted_cands[:min(40, len(sorted_cands))]

            # Find a compatible pair
            found_pair = False
            for i in range(len(sorted_cands)):
                c1 = sorted_cands[i]
                for j in range(i + 1, len(sorted_cands)):
                    c2 = sorted_cands[j]
                    if conflict[c1, c2] == 0:
                        # Do the swap
                        selected.remove(s)
                        sel_set.discard(s)
                        selected.append(c1)
                        sel_set.add(c1)
                        selected.append(c2)
                        sel_set.add(c2)
                        found_pair = True
                        improved = True
                        break
                if found_pair:
                    break

            if found_pair:
                break

        if not improved:
            break

    # ---- Phase 4: Final greedy fill ----
    if time.time() < deadline - 2.0:
        if selected:
            compat_mask = np.ones(n, dtype=np.uint8)
            for s in selected:
                compat_mask = compat_mask & compat[s]
            for s in selected:
                compat_mask[s] = 0
        else:
            compat_mask = np.ones(n, dtype=np.uint8)

        remaining = compat_mask.copy()
        while time.time() < deadline - 1.0:
            addable = np.where(remaining > 0)[0]
            if len(addable) == 0:
                break
            deg = (conflict[addable] * remaining[np.newaxis, :]).sum(axis=1).astype(np.int64)
            min_idx = np.argmin(deg)
            new_idx = addable[min_idx]
            selected.append(new_idx)
            remaining = remaining & (1 - conflict[new_idx])
            remaining[new_idx] = 0

    if not selected:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)

    result = candidates[np.array(selected, dtype=np.int64)]
    return result.astype(np.int64)
