import itertools
import numpy as np
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s kill

    # Precompute t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    t_sub_to_idx = {s: i for i, s in enumerate(t_subsets)}

    # Precompute all k-subsets and their t-subset memberships as bitmasks
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    n_t_in_k = len(list(itertools.combinations(range(k), t)))

    # For each k-subset, compute bitmask of t-subsets it covers
    # n_t = C(19,5) = 11628, fits in 2 numpy uint64
    n_words = (n_t + 63) // 64
    block_masks = np.zeros((n_k, n_words), dtype=np.uint64)

    for i, block in enumerate(k_subsets):
        for t_combo in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_combo]
            word_idx = idx >> 6
            bit_idx = idx & 63
            block_masks[i, word_idx] |= (1 << bit_idx)

    # Full mask: all t-subsets
    full_mask = np.zeros(n_words, dtype=np.uint64)
    for w in range(n_words):
        full_mask[w] = 0xFFFFFFFFFFFFFFFF
    # Mask out bits beyond n_t in the last word
    last_word_bits = n_t & 63
    if last_word_bits:
        full_mask[-1] = (1 << last_word_bits) - 1

    def count_set_bits(mask):
        """Count set bits in a multi-word bitmask."""
        total = 0
        for w in range(n_words):
            val = mask[w]
            while val:
                val &= val - 1
                total += 1
        return total

    def popcount_u64(val):
        """Popcount for a single uint64."""
        val = val - ((val >> 1) & 0x5555555555555555)
        val = (val & 0x3333333333333333) + ((val >> 2) & 0x3333333333333333)
        val = (val + (val >> 4)) & 0x0F0F0F0F0F0F0F0F
        return int(((val * 0x0101010101010101) >> 56) & 0xFF)

    def mask_count_set(mask):
        total = 0
        for w in range(n_words):
            total += popcount_u64(int(mask[w]))
        return total

    def greedy_cover(weights, rng):
        """Weighted greedy set cover. Returns (selected_indices, uncovered_mask)."""
        uncovered = full_mask.copy()
        selected = []
        selected_set = set()

        while True:
            # Check if all covered
            all_covered = True
            for w in range(n_words):
                if uncovered[w]:
                    all_covered = False
                    break
            if all_covered:
                break

            # Compute weighted gain for each block
            # gain[i] = sum of weights of t-subsets in block i that are uncovered
            best_gain = -1
            best_candidates = []

            for i in range(n_k):
                if i in selected_set:
                    continue
                # Compute intersection of block_masks[i] and uncovered
                gain = 0
                for w in range(n_words):
                    inter = int(block_masks[i, w]) & int(uncovered[w])
                    if inter:
                        gain += popcount_u64(inter) * (weights[w] if isinstance(weights, np.ndarray) and len(weights) == n_words else 1)
                # Actually weights should be per t-subset, not per word. Let me redo.
                # For simplicity with word-level, use uniform weight per bit = 1
                # But for Lagrangian, we need per-t-subset weights.
                # Let's just use the count of uncovered bits covered.
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)

            if best_gain <= 0:
                break

            # Pick randomly among best candidates
            chosen = rng.choice(best_candidates)
            selected.append(chosen)
            selected_set.add(chosen)

            # Update uncovered
            for w in range(n_words):
                uncovered[w] = uncovered[w] & ~block_masks[chosen, w]

        return selected, uncovered

    def greedy_cover_fast(uncovered, selected_set, rng):
        """Faster greedy: precompute gains using numpy where possible."""
        selected = []

        while True:
            all_covered = True
            for w in range(n_words):
                if uncovered[w]:
                    all_covered = False
                    break
            if all_covered:
                break

            # Compute gain for each block: count of uncovered bits it covers
            # Use numpy for the AND and then count
            # Convert uncovered to list of ints
            unc_list = [int(uncovered[w]) for w in range(n_words)]

            best_gain = -1
            best_candidates = []

            for i in range(n_k):
                if i in selected_set:
                    continue
                gain = 0
                for w in range(n_words):
                    inter = int(block_masks[i, w]) & unc_list[w]
                    if inter:
                        gain += popcount_u64(inter)
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)

            if best_gain <= 0:
                break

            chosen = rng.choice(best_candidates)
            selected.append(chosen)
            selected_set.add(chosen)

            for w in range(n_words):
                uncovered[w] = uncovered[w] & ~block_masks[chosen, w]

        return selected

    def remove_redundant(selected, deadline):
        """Remove redundant blocks iteratively."""
        # Build coverage count
        coverage = np.zeros(n_t, dtype=np.int32)
        for bi in selected:
            for w in range(n_words):
                bits = int(block_masks[bi, w])
                while bits:
                    bit = bits & (-bits)
                    bit_idx = bit.bit_length() - 1
                    global_idx = w * 64 + bit_idx
                    if global_idx < n_t:
                        coverage[global_idx] += 1
                    bits ^= bit

        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                bi = selected[i]
                # Check if all t-subsets in this block have coverage >= 2
                redundant = True
                for w in range(n_words):
                    bits = int(block_masks[bi, w])
                    while bits:
                        bit = bits & (-bits)
                        bit_idx = bit.bit_length() - 1
                        global_idx = w * 64 + bit_idx
                        if global_idx < n_t:
                            if coverage[global_idx] <= 1:
                                redundant = False
                                break
                        bits ^= bit
                    if not redundant:
                        break

                if redundant:
                    # Remove block
                    for w in range(n_words):
                        bits = int(block_masks[bi, w])
                        while bits:
                            bit = bits & (-bits)
                            bit_idx = bit.bit_length() - 1
                            global_idx = w * 64 + bit_idx
                            if global_idx < n_t:
                                coverage[global_idx] -= 1
                            bits ^= bit
                    selected.pop(i)
                    changed = True
                    break

        return selected

    def swap_improve(selected, deadline):
        """Try to swap blocks to reduce count or maintain count with better coverage."""
        if len(selected) < 2:
            return selected

        # Build coverage
        coverage = np.zeros(n_t, dtype=np.int32)
        for bi in selected:
            for w in range(n_words):
                bits = int(block_masks[bi, w])
                while bits:
                    bit = bits & (-bits)
                    bit_idx = bit.bit_length() - 1
                    global_idx = w * 64 + bit_idx
                    if global_idx < n_t:
                        coverage[global_idx] += 1
                    bits ^= bit

        selected_set = set(selected)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each block and see if we can cover the gap with one other block
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                bi = selected[i]
                # What t-subsets would become uncovered?
                gap = np.zeros(n_t, dtype=bool)
                for w in range(n_words):
                    bits = int(block_masks[bi, w])
                    while bits:
                        bit = bits & (-bits)
                        bit_idx = bit.bit_length() - 1
                        global_idx = w * 64 + bit_idx
                        if global_idx < n_t:
                            if coverage[global_idx] == 1:
                                gap[global_idx] = True
                        bits ^= bit

                if not np.any(gap):
                    # Block is redundant
                    for w in range(n_words):
                        bits = int(block_masks[bi, w])
                        while bits:
                            bit = bits & (-bits)
                            bit_idx = bit.bit_length() - 1
                            global_idx = w * 64 + bit_idx
                            if global_idx < n_t:
                                coverage[global_idx] -= 1
                            bits ^= bit
                    selected.pop(i)
                    selected_set.discard(bi)
                    improved = True
                    break

                # Try to find a single block not in selected that covers all gap
                # gap_count = number of gap t-subsets
                gap_count = int(np.sum(gap))
                if gap_count > n_t_in_k:
                    continue

                # Find candidate blocks that cover all gap t-subsets
                # For efficiency, check blocks that have high overlap with gap
                best_candidate = -1
                best_extra = 0
                for j in range(n_k):
                    if j in selected_set:
                        continue
                    # Check if block j covers all gap
                    covers_all = True
                    for w in range(n_words):
                        bits = int(block_masks[j, w])
                        # Need to check if all gap bits in this word are in block j
                        # gap in word w: we need gap_mask[w] & ~block_masks[j,w] == 0
                        # But we don't have gap as bitmask per word easily
                        pass
                    # This is getting complex. Skip swap for now.
                    break
                # Skip swap improvement for time
                break

        return selected

    # Main: try multiple strategies
    best_solution = None
    best_count = float('inf')

    # Strategy 1: Pure greedy with random tie-breaking (multiple restarts)
    rng = random.Random(42)

    # Time budget for greedy runs
    greedy_budget = 60.0
    redundancy_budget = 20.0

    n_restarts = 5

    for restart in range(n_restarts):
        if time.time() - start_time > greedy_budget:
            break

        rng = random.Random(42 + restart)
        selected = []
        selected_set = set()
        uncovered = full_mask.copy()

        while True:
            if time.time() - start_time > greedy_budget:
                break

            all_covered = True
            for w in range(n_words):
                if uncovered[w]:
                    all_covered = False
                    break
            if all_covered:
                break

            # Compute gains
            unc_list = [int(uncovered[w]) for w in range(n_words)]
            best_gain = -1
            best_candidates = []

            for i in range(n_k):
                if i in selected_set:
                    continue
                gain = 0
                for w in range(n_words):
                    inter = int(block_masks[i, w]) & unc_list[w]
                    if inter:
                        gain += popcount_u64(inter)
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)

            if best_gain <= 0:
                break

            # Weighted random selection among top candidates
            # Use softmax-like: favor higher gain
            if len(best_candidates) == 1:
                chosen = best_candidates[0]
            else:
                # Add small random perturbation
                gains_for_cands = []
                for c in best_candidates:
                    g = 0
                    for w in range(n_words):
                        inter = int(block_masks[c, w]) & unc_list[w]
                        if inter:
                            g += popcount_u64(inter)
                    gains_for_cands.append(g)
                # All have same gain, so pick randomly
                chosen = rng.choice(best_candidates)

            selected.append(chosen)
            selected_set.add(chosen)

            for w in range(n_words):
                uncovered[w] = uncovered[w] & ~block_masks[chosen, w]

        # Check if complete
        complete = True
        for w in range(n_words):
            if uncovered[w]:
                complete = False
                break

        if not complete:
            # Try to complete with remaining time
            remaining_deadline = start_time + greedy_budget
            while True:
                if time.time() > remaining_deadline:
                    break
                all_covered = True
                for w in range(n_words):
                    if uncovered[w]:
                        all_covered = False
                        break
                if all_covered:
                    break
                unc_list = [int(uncovered[w]) for w in range(n_words)]
                best_gain = -1
                best_candidates = []
                for i in range(n_k):
                    if i in selected_set:
                        continue
                    gain = 0
                    for w in range(n_words):
                        inter = int(block_masks[i, w]) & unc_list[w]
                        if inter:
                            gain += popcount_u64(inter)
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [i]
                    elif gain == best_gain and gain > 0:
                        best_candidates.append(i)
                if best_gain <= 0:
                    break
                chosen = rng.choice(best_candidates)
                selected.append(chosen)
                selected_set.add(chosen)
                for w in range(n_words):
                    uncovered[w] = uncovered[w] & ~block_masks[chosen, w]

        if len(selected) < best_count:
            # Redundancy removal
            red_deadline = min(time.time() + redundancy_budget, start_time + 110.0)
            selected = remove_redundant(selected, red_deadline)
            best_count = len(selected)
            best_solution = selected[:]

    # Final: try one more round of redundancy removal on best
    if best_solution is not None:
        red_deadline = min(time.time() + 15.0, start_time + 115.0)
        best_solution = remove_redundant(best_solution, red_deadline)

    if best_solution is None:
        # Fallback: should not happen
        return []

    return [list(k_subsets[i]) for i in best_solution]
