"""Kissing number in dimension 11: simulated annealing on conflict graph."""
import numpy as np
import itertools
import time
import random


def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    D = 11

    # Generate candidate vectors
    candidates_list = []

    # Weight-4 vectors (norm^2=4): C(11,4)*16 = 5280
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Weight-3 vectors (norm^2=3): C(11,3)*8 = 1320
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Weight-2 vectors (norm^2=2): C(11,2)*4 = 220
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Weight-1 vectors (norm^2=1): 22
    for pos in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[pos] = s
            candidates_list.append(v)

    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)

    # Precompute norms squared
    norms_sq = np.sum(candidates.astype(np.int32) ** 2, axis=1)
    max_norm_sq = int(norms_sq.max())  # 4

    # Precompute dot product matrix
    dot = candidates.astype(np.int32) @ candidates.T.astype(np.int32)

    # Constraint: ||v_i - v_j||^2 >= max_norm_sq
    threshold = (norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - max_norm_sq) / 2.0
    compat = (dot <= threshold).astype(np.uint8)
    np.fill_diagonal(compat, 1)

    # conflict[i,j] = 1 if i and j CANNOT both be selected
    conflict = (1 - compat).astype(np.uint8)
    np.fill_diagonal(conflict, 0)

    # Precompute full degree
    full_deg = conflict.sum(axis=1)

    # Use bitsets for fast compatibility checks
    # Each row of compat_bits is a list of 64-bit integers representing the compatibility set
    n_words = (n + 63) // 64
    compat_bits = np.zeros((n, n_words), dtype=np.uint64)
    for i in range(n):
        row = compat[i]
        for w in range(n_words):
            bits = 0
            for b in range(64):
                idx = w * 64 + b
                if idx < n and row[idx]:
                    bits |= (1 << b)
            compat_bits[i, w] = bits

    def bits_popcount(bits):
        """Count set bits in a uint64."""
        count = 0
        while bits:
            bits &= bits - 1
            count += 1
        return count

    def bits_and(a, b):
        """Element-wise AND of two bitset arrays."""
        return a & b

    def bits_or(a, b):
        """Element-wise OR of two bitset arrays."""
        return a | b

    def bits_not(a):
        """Complement of bitset (mask out bits beyond n)."""
        result = ~a
        # Mask off bits beyond n in the last word
        last_word_bits = n % 64
        if last_word_bits == 0:
            last_word_bits = 64
        if n_words > 0:
            mask = (1 << last_word_bits) - 1
            result[-1] &= mask
        return result

    def bits_count(bits_arr):
        """Count total set bits in a bitset array."""
        total = 0
        for w in range(n_words):
            total += bits_popcount(bits_arr[w])
        return total

    def greedy_initial_solution(time_limit):
        """Generate an initial solution using min-degree greedy."""
        remaining = np.ones(n, dtype=np.uint8)
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

    def simulated_annealing(initial_selected, time_limit, max_iters=200000):
        """Simulated annealing to improve the independent set."""
        selected = list(initial_selected)
        sel_set = set(selected)
        size = len(selected)

        # Build compatibility mask: which nodes are compatible with ALL selected
        # Using bitsets for speed
        compat_mask = np.ones(n_words, dtype=np.uint64)
        for s in selected:
            compat_mask = bits_and(compat_mask, compat_bits[s])
        # Remove selected nodes from mask
        for s in selected:
            word = s // 64
            bit = s % 64
            compat_mask[word] &= ~(1 << bit)

        best_size = size
        best_selected = list(selected)

        T_start = 1.0
        T_end = 0.01
        cooling_rate = (T_end / T_start) ** (1.0 / max(1, max_iters))

        T = T_start
        it = 0
        rng = np.random.RandomState(42)

        while it < max_iters and time.time() < time_limit:
            T *= cooling_rate
            it += 1

            # Randomly pick a move type
            move_type = rng.randint(0, 3)

            if move_type == 0:
                # 1-for-1 swap: remove one selected, add one unselected
                if size == 0:
                    continue
                s = selected[rng.randint(0, size - 1)]
                # Find candidates compatible with all selected except s
                mask_except = np.ones(n_words, dtype=np.uint64)
                for s2 in selected:
                    if s2 != s:
                        mask_except = bits_and(mask_except, compat_bits[s2])
                # Remove all selected
                for s2 in selected:
                    word = s2 // 64
                    bit = s2 % 64
                    mask_except[word] &= ~(1 << bit)

                # Get candidate indices
                cands = []
                for w in range(n_words):
                    bits = mask_except[w]
                    while bits:
                        lb = bits & (-bits)
                        b = lb.bit_length() - 1
                        idx = w * 64 + b
                        if idx < n:
                            cands.append(idx)
                        bits ^= lb

                if not cands:
                    continue

                c = cands[rng.randint(0, len(cands) - 1)]

                # Perform swap
                selected.remove(s)
                sel_set.discard(s)
                selected.append(c)
                sel_set.add(c)
                size = len(selected)

                # Update compat_mask
                compat_mask = np.ones(n_words, dtype=np.uint64)
                for s2 in selected:
                    compat_mask = bits_and(compat_mask, compat_bits[s2])
                for s2 in selected:
                    word = s2 // 64
                    bit = s2 % 64
                    compat_mask[word] &= ~(1 << bit)

                # Accept or reject based on energy change (here, size is the objective)
                # Since we always keep the same size for 1-for-1, accept with probability based on T
                # Actually, for 1-for-1 the size doesn't change, so we need a different criterion
                # Let's use: accept if the new node has lower degree (less constraining)
                if full_deg[c] > full_deg[s] and rng.random() > T:
                    # Reject: revert
                    selected.remove(c)
                    sel_set.discard(c)
                    selected.append(s)
                    sel_set.add(s)
                    compat_mask = np.ones(n_words, dtype=np.uint64)
                    for s2 in selected:
                        compat_mask = bits_and(compat_mask, compat_bits[s2])
                    for s2 in selected:
                        word = s2 // 64
                        bit = s2 % 64
                        compat_mask[word] &= ~(1 << bit)

            elif move_type == 1:
                # Add one node if possible
                addable = []
                for w in range(n_words):
                    bits = compat_mask[w]
                    while bits:
                        lb = bits & (-bits)
                        b = lb.bit_length() - 1
                        idx = w * 64 + b
                        if idx < n:
                            addable.append(idx)
                        bits ^= lb

                if not addable:
                    continue

                c = addable[rng.randint(0, len(addable) - 1)]
                selected.append(c)
                sel_set.add(c)
                size += 1
                compat_mask = bits_and(compat_mask, compat_bits[c])
                word = c // 64
                bit = c % 64
                compat_mask[word] &= ~(1 << bit)

                if size > best_size:
                    best_size = size
                    best_selected = list(selected)

            elif move_type == 2:
                # Remove one node (to allow future additions)
                if size <= 5:
                    continue
                s = selected[rng.randint(0, size - 1)]
                selected.remove(s)
                sel_set.discard(s)
                size -= 1

                # Update compat_mask
                compat_mask = np.ones(n_words, dtype=np.uint64)
                for s2 in selected:
                    compat_mask = bits_and(compat_mask, compat_bits[s2])
                for s2 in selected:
                    word = s2 // 64
                    bit = s2 % 64
                    compat_mask[word] &= ~(1 << bit)

            else:
                # 1-for-2: remove one, add two
                if size == 0:
                    continue
                s = selected[rng.randint(0, size - 1)]
                mask_except = np.ones(n_words, dtype=np.uint64)
                for s2 in selected:
                    if s2 != s:
                        mask_except = bits_and(mask_except, compat_bits[s2])
                for s2 in selected:
                    word = s2 // 64
                    bit = s2 % 64
                    mask_except[word] &= ~(1 << bit)

                cands = []
                for w in range(n_words):
                    bits = mask_except[w]
                    while bits:
                        lb = bits & (-bits)
                        b = lb.bit_length() - 1
                        idx = w * 64 + b
                        if idx < n:
                            cands.append(idx)
                        bits ^= lb

                if len(cands) < 2:
                    continue

                # Pick two compatible candidates
                c1 = cands[rng.randint(0, len(cands) - 1)]
                # Find c2 compatible with c1 and all in mask_except
                mask_with_c1 = bits_and(mask_except, compat_bits[c1])
                # Remove c1
                word1 = c1 // 64
                bit1 = c1 % 64
                mask_with_c1[word1] &= ~(1 << bit1)

                cands2 = []
                for w in range(n_words):
                    bits = mask_with_c1[w]
                    while bits:
                        lb = bits & (-bits)
                        b = lb.bit_length() - 1
                        idx = w * 64 + b
                        if idx < n:
                            cands2.append(idx)
                        bits ^= lb

                if not cands2:
                    continue

                c2 = cands2[rng.randint(0, len(cands2) - 1)]

                # Perform 1-for-2
                selected.remove(s)
                sel_set.discard(s)
                selected.append(c1)
                sel_set.add(c1)
                selected.append(c2)
                sel_set.add(c2)
                size += 1

                compat_mask = np.ones(n_words, dtype=np.uint64)
                for s2 in selected:
                    compat_mask = bits_and(compat_mask, compat_bits[s2])
                for s2 in selected:
                    word = s2 // 64
                    bit = s2 % 64
                    compat_mask[word] &= ~(1 << bit)

                if size > best_size:
                    best_size = size
                    best_selected = list(selected)

        return best_selected

    # Main search
    best_selected = []
    np.random.seed(42)
    random.seed(42)

    # Generate multiple initial solutions and run simulated annealing on each
    n_restarts = 0
    while time.time() < deadline - 20.0:
        n_restarts += 1

        # Generate initial solution
        if n_restarts % 5 == 0:
            # Use a randomized greedy
            frac = np.random.uniform(0.6, 1.0)
            subset_size = int(n * frac)
            subset_indices = np.random.choice(n, size=subset_size, replace=False)
            remaining_init = np.zeros(n, dtype=np.uint8)
            remaining_init[subset_indices] = 1

            # Min-degree greedy on subset
            remaining = remaining_init.copy()
            selected = []
            while remaining.any():
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
        else:
            # Standard min-degree greedy
            selected = greedy_initial_solution(deadline - 20.0)

        if len(selected) > len(best_selected):
            best_selected = list(selected)

        # Run simulated annealing
        sa_time_limit = deadline - 20.0
        improved = simulated_annealing(selected, sa_time_limit, max_iters=100000)

        if len(improved) > len(best_selected):
            best_selected = improved

        # Check if we've run out of time for more restarts
        if time.time() > deadline - 20.0:
            break

    # Final greedy fill
    if best_selected and time.time() < deadline - 2.0:
        sel_set = set(best_selected)
        compat_with_all = np.ones(n, dtype=np.uint8)
        for s in best_selected:
            compat_with_all = compat_with_all & compat[s]
        for s in best_selected:
            compat_with_all[s] = 0
        addable = np.where(compat_with_all > 0)[0]
        for idx in addable[np.argsort(full_deg[addable])]:
            if time.time() > deadline - 2.0:
                break
            best_selected.append(idx)
            sel_set.add(idx)
            compat_with_all = compat_with_all & compat[idx]
            compat_with_all[idx] = 0

    if not best_selected:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)

    result = candidates[np.array(best_selected, dtype=np.int64)]
    return result.astype(np.int64)
