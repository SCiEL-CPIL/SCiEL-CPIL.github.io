"""Covering code K_3(7,1) via greedy set cover with numpy-accelerated coverage counting."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

_DEADLINE = 140.0


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + _DEADLINE

    total = q ** n
    powers = [q ** (n - 1 - i) for i in range(n)]

    # Precompute all digit tuples
    all_words = []
    for idx in range(total):
        digits = []
        x = idx
        for _ in range(n):
            digits.append(x % q)
            x //= q
        digits.reverse()
        all_words.append(tuple(digits))

    # Precompute neighbor offsets for distance 1
    # For each position, for each alternative value, the index offset
    neighbor_offsets = []
    for pos in range(n):
        for alt in range(q):
            # We'll compute the actual neighbor index on the fly
            pass

    # Precompute ball masks using numpy for vectorized computation
    # For each center index, compute the set of indices within distance <= radius
    # Ball size for q=3, n=7, r=1: 1 + 7*2 = 15
    ball_masks = np.zeros(total, dtype=object)

    # Use a more efficient approach: precompute for each position and value,
    # which indices have that value at that position
    # Then for a center, its ball = {center} ∪ ⋃_{pos, alt≠center[pos]} {indices with alt at pos}
    
    # Actually, let's just precompute the neighbor indices for each center
    # For 2187 centers, this is manageable
    neighbor_indices = [[] for _ in range(total)]
    for center_idx in range(total):
        center = all_words[center_idx]
        for pos in range(n):
            cur_val = center[pos]
            for alt in range(q):
                if alt == cur_val:
                    continue
                new_idx = center_idx + (alt - cur_val) * powers[pos]
                neighbor_indices[center_idx].append(new_idx)

    # Build ball masks as Python ints (bitsets)
    ball_masks_list = [0] * total
    for center_idx in range(total):
        mask = 1 << center_idx
        for nb in neighbor_indices[center_idx]:
            mask |= (1 << nb)
        ball_masks_list[center_idx] = mask

    # Greedy set cover
    all_mask = (1 << total) - 1
    uncovered = all_mask
    selected = []
    remaining_indices = list(range(total))

    while uncovered != 0:
        if time.time() > deadline - 15:
            break

        best_count = -1
        best_idx = -1
        best_pos_in_rem = -1

        # Iterate through remaining candidates to find best coverage
        for pos in range(len(remaining_indices)):
            c = remaining_indices[pos]
            covered = ball_masks_list[c] & uncovered
            cnt = covered.bit_count()
            if cnt > best_count:
                best_count = cnt
                best_idx = c
                best_pos_in_rem = pos

        if best_idx == -1 or best_count == 0:
            break

        selected.append(best_idx)
        uncovered &= ~ball_masks_list[best_idx]

        # Remove best_idx from remaining_indices
        last_pos = len(remaining_indices) - 1
        remaining_indices[best_pos_in_rem] = remaining_indices[last_pos]
        remaining_indices.pop()

    # If greedy didn't finish, fall back to direct sum construction
    if uncovered != 0:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Local search: remove redundant codewords
    # Use coverage counting for efficiency
    def try_remove_redundant(selected, ball_masks_list, deadline):
        removed_any = False
        n_sel = len(selected)
        if n_sel == 0:
            return selected, removed_any

        # Build coverage count: for each word, how many selected codewords cover it
        # Use a list of sets for efficiency
        covered_by = [[] for _ in range(total)]
        for i, c in enumerate(selected):
            mask = ball_masks_list[c]
            # Extract set bits
            m = mask
            while m:
                low_bit = m & (-m)
                idx = low_bit.bit_length() - 1
                covered_by[idx].append(i)
                m ^= low_bit

        # Try removing each codeword
        for i in range(n_sel - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            # Check if any word is only covered by selected[i]
            redundant = True
            mask = ball_masks_list[selected[i]]
            m = mask
            while m:
                low_bit = m & (-m)
                idx = low_bit.bit_length() - 1
                if len(covered_by[idx]) == 1 and covered_by[idx][0] == i:
                    redundant = False
                    break
                m ^= low_bit

            if redundant:
                # Remove selected[i]
                selected.pop(i)
                # Update covered_by
                mask = ball_masks_list[selected[i]]
                m = mask
                while m:
                    low_bit = m & (-m)
                    idx = low_bit.bit_length() - 1
                    covered_by[idx].remove(i)
                    m ^= low_bit
                removed_any = True
                break

        return selected, removed_any

    # Multiple passes of removal
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        selected, removed_any = try_remove_redundant(selected, ball_masks_list, deadline)
        if not removed_any:
            break

    # Local search: try swaps to reduce size
    # Try to replace a codeword with one that covers more uncovered words
    best_solution = selected[:]
    best_size = len(selected)

    # Try several random restarts with different orderings
    if time.time() < deadline - 25:
        for restart in range(8):
            if time.time() > deadline - 15:
                break
            random.seed(42 + restart * 7)
            shuffled = list(range(total))
            random.shuffle(shuffled)

            uncovered_r = all_mask
            selected_r = []
            remaining_r_indices = list(shuffled)

            while uncovered_r != 0:
                if time.time() > deadline - 15:
                    break
                best_count = -1
                best_idx = -1
                best_pos_in_rem = -1

                for pos in range(len(remaining_r_indices)):
                    c = remaining_r_indices[pos]
                    covered = ball_masks_list[c] & uncovered_r
                    cnt = covered.bit_count()
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = c
                        best_pos_in_rem = pos

                if best_idx == -1 or best_count == 0:
                    break
                selected_r.append(best_idx)
                uncovered_r &= ~ball_masks_list[best_idx]

                last_pos = len(remaining_r_indices) - 1
                remaining_r_indices[best_pos_in_rem] = remaining_r_indices[last_pos]
                remaining_r_indices.pop()

            if uncovered_r == 0 and len(selected_r) < best_size:
                # Run redundancy removal
                for _ in range(5):
                    if time.time() > deadline - 10:
                        break
                    selected_r, removed_any = try_remove_redundant(selected_r, ball_masks_list, deadline)
                    if not removed_any:
                        break
                if len(selected_r) < best_size:
                    best_solution = selected_r
                    best_size = len(selected_r)

    # Convert indices back to digit lists
    result = []
    for idx in best_solution:
        digits = []
        x = idx
        for _ in range(n):
            digits.append(x % q)
            x //= q
        digits.reverse()
        result.append(digits)

    return result
# EVOLVE-BLOCK-END
