import numpy as np
import itertools
import time
import random

D = 11

def construct():
    deadline = time.time() + 115  # safety margin before 120s kill

    # Generate all 4-subsets of range(11)
    subsets = list(itertools.combinations(range(D), 4))
    n = len(subsets)  # 330

    # Precompute: for each pair of subsets, whether they are compatible
    # Compatible if |S ∩ T| <= 2 (i.e., they don't share 3 or 4 elements)

    # Build compatibility matrix using vectorized approach
    masks = np.zeros(n, dtype=np.int64)
    for i, s in enumerate(subsets):
        m = 0
        for pos in s:
            m |= (1 << pos)
        masks[i] = m

    # Compute pairwise intersection sizes
    m1 = masks[:, np.newaxis]  # (n, 1)
    m2 = masks[np.newaxis, :]  # (1, n)
    and_matrix = m1 & m2

    # Count bits: use lookup table
    popcount_table = np.zeros(2048, dtype=np.int32)
    for i in range(2048):
        popcount_table[i] = bin(i).count('1')

    popcount_matrix = popcount_table[and_matrix]  # (n, n)
    compatible = (popcount_matrix <= 2)  # (n, n) boolean matrix

    # Conflict count for each subset
    conflict_count = np.sum(~compatible, axis=1)

    # Helper: check if a set of selected indices is all pairwise compatible
    def is_valid_selection(sel_indices):
        if len(sel_indices) <= 1:
            return True
        sub_compat = compatible[np.ix_(sel_indices, sel_indices)]
        iu = np.triu_indices(len(sel_indices), k=1)
        return np.all(sub_compat[iu])

    # Helper: compute number of conflicts for a boolean selection vector
    def count_conflicts(x):
        sel = np.where(x)[0]
        if len(sel) <= 1:
            return 0
        sub_compat = compatible[np.ix_(sel, sel)]
        iu = np.triu_indices(len(sel), k=1)
        return np.sum(~sub_compat[iu])

    # Greedy initialization with randomization
    rng = np.random.default_rng(42)

    best_selected = None
    best_score = 0

    # Multiple greedy restarts with different orderings
    for restart in range(5):
        if time.time() > deadline - 5:
            break

        # Shuffle order for randomization
        order = np.arange(n)
        if restart > 0:
            rng.shuffle(order)

        remaining = np.ones(n, dtype=bool)
        selected = np.zeros(n, dtype=bool)

        while remaining.any():
            candidates = np.where(remaining)[0]
            if len(candidates) == 0:
                break
            # Pick the one with minimum conflicts among remaining
            conflicts_with_remaining = np.sum(np.where(remaining[None, :], ~compatible, 0), axis=1)
            min_idx = candidates[np.argmin(conflicts_with_remaining[candidates])]
            selected[min_idx] = True
            remaining[min_idx] = False
            remaining[~compatible[min_idx, :]] = False

        score = np.sum(selected)
        if score > best_score:
            best_score = score
            best_selected = selected.copy()

    current = best_selected.copy()
    current_score = best_score

    # Local search with 1-flip, 1-swap, and 2-swap moves
    T = 10.0
    cooling = 0.9995

    while time.time() < deadline - 2:
        # Check time more frequently
        move_type = rng.choice(['flip_in', 'flip_out', 'swap1', 'swap2'], p=[0.3, 0.15, 0.25, 0.3])

        if move_type == 'flip_in':
            unselected = np.where(~current)[0]
            if len(unselected) == 0:
                continue
            # Try multiple candidates, pick best valid one
            candidates = rng.choice(unselected, size=min(10, len(unselected)), replace=False)
            for idx in candidates:
                new_x = current.copy()
                new_x[idx] = True
                # Check if idx is compatible with all currently selected
                sel = np.where(current)[0]
                if len(sel) == 0 or np.all(compatible[idx, sel]):
                    current = new_x
                    current_score += 1
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = current.copy()
                    break
        elif move_type == 'flip_out':
            selected_idx = np.where(current)[0]
            if len(selected_idx) == 0:
                continue
            idx = rng.choice(selected_idx)
            current[idx] = False
            current_score -= 1
        elif move_type == 'swap1':
            selected_idx = np.where(current)[0]
            unselected = np.where(~current)[0]
            if len(selected_idx) == 0 or len(unselected) == 0:
                continue
            # Try multiple swaps
            out_candidates = rng.choice(selected_idx, size=min(5, len(selected_idx)), replace=False)
            in_candidates = rng.choice(unselected, size=min(5, len(unselected)), replace=False)
            found = False
            for out_idx in out_candidates:
                for in_idx in in_candidates:
                    # Check if in_idx is compatible with all selected except out_idx
                    sel_other = selected_idx[selected_idx != out_idx]
                    if len(sel_other) == 0 or np.all(compatible[in_idx, sel_other]):
                        current[out_idx] = False
                        current[in_idx] = True
                        # Score stays same
                        if current_score > best_score:
                            best_score = current_score
                            best_selected = current.copy()
                        found = True
                        break
                if found:
                    break
        elif move_type == 'swap2':
            # Remove 2 selected, add 2 unselected
            selected_idx = np.where(current)[0]
            unselected = np.where(~current)[0]
            if len(selected_idx) < 2 or len(unselected) < 2:
                continue
            # Try a few random 2-swap combinations
            out_candidates = rng.choice(selected_idx, size=min(4, len(selected_idx)), replace=False)
            in_candidates = rng.choice(unselected, size=min(4, len(unselected)), replace=False)
            found = False
            for i in range(len(out_candidates)):
                for j in range(i + 1, len(out_candidates)):
                    out1, out2 = out_candidates[i], out_candidates[j]
                    for k in range(len(in_candidates)):
                        for l in range(k + 1, len(in_candidates)):
                            in1, in2 = in_candidates[k], in_candidates[l]
                            # Check: in1 and in2 compatible with each other
                            if not compatible[in1, in2]:
                                continue
                            # Check: in1 and in2 compatible with all selected except out1, out2
                            sel_other = selected_idx[(selected_idx != out1) & (selected_idx != out2)]
                            if len(sel_other) > 0:
                                if not (np.all(compatible[in1, sel_other]) and np.all(compatible[in2, sel_other])):
                                    continue
                            # Valid 2-swap
                            current[out1] = False
                            current[out2] = False
                            current[in1] = True
                            current[in2] = True
                            # Score stays same (removed 2, added 2)
                            if current_score > best_score:
                                best_score = current_score
                                best_selected = current.copy()
                            found = True
                            break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break

        T *= cooling

    # Final greedy fill pass: try to add any remaining compatible vector
    final_selected = best_selected.copy()
    final_score = np.sum(final_selected)
    sel = np.where(final_selected)[0]

    for idx in range(n):
        if final_selected[idx]:
            continue
        # Check if idx is compatible with all currently selected
        if len(sel) == 0:
            final_selected[idx] = True
            sel = np.append(sel, idx)
            final_score += 1
        else:
            if np.all(compatible[idx, sel]):
                final_selected[idx] = True
                sel = np.append(sel, idx)
                final_score += 1

    # Build vectors: for each selected subset, generate all 2^4 = 16 sign combinations
    sel_indices = np.where(final_selected)[0]

    vectors = []
    for idx in sel_indices:
        s = subsets[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(s, signs):
                v[pos] = sign
            vectors.append(v)

    if len(vectors) == 0:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

    return np.array(vectors, dtype=np.int64)
