import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    total_space = q ** n

    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Precompute neighbor bitmasks
    # masks[i] is a bitmask where bit j is set if word j is within Hamming distance radius of word i
    masks = [0] * total_space
    for i in range(total_space):
        m = 1 << i
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                j = i + diff * powers[pos]
                m |= (1 << j)
        masks[i] = m

    full_mask = (1 << total_space) - 1

    best_size = total_space
    best_selected = list(range(total_space))

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    def polish(selected, time_limit):
        """Remove redundant codewords iteratively."""
        selected = selected[:]
        if not selected:
            return selected

        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            mm = masks[w]
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit

        changed = True
        while changed and time.time() < time_limit:
            changed = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                m = masks[w]

                removable = True
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        removable = False
                        break
                    mm ^= lowbit

                if removable:
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    selected.pop(idx)
                    changed = True

        return selected

    def local_search(selected, time_limit):
        """
        Local search with 1-for-1 swaps, 2-for-0 deletions, and 2-for-1 swaps.
        """
        selected = selected[:]
        if not selected:
            return selected

        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            mm = masks[w]
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit

        selected_set = set(selected)

        # 1. Polish again to ensure no simple deletions remain
        selected = polish(selected, time_limit)
        if len(selected) == 0:
            return selected
        
        # Rebuild counts after polish
        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            mm = masks[w]
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit
        selected_set = set(selected)

        # 2. Local Search Loop
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            # 1-for-1 swaps and deletions
            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                m = masks[w]

                # Check for deletion (unique points count is 0)
                unique_mask = 0
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        unique_mask |= lowbit
                    mm ^= lowbit

                if unique_mask == 0:
                    # Delete w
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    selected.pop(idx)
                    selected_set.remove(w)
                    improved = True
                    continue

                # Check for 1-for-1 swap
                best_gain = -1
                best_word = -1
                for w2 in range(total_space):
                    if w2 in selected_set:
                        continue
                    m2 = masks[w2]
                    # w2 must cover all unique points of w
                    if (m2 & unique_mask) == unique_mask:
                        gain = (m2 & ~m).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_word = w2

                if best_word >= 0 and best_gain > 0:
                    # Swap w with best_word
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit

                    mm = masks[best_word]
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] += 1
                        mm ^= lowbit

                    selected[idx] = best_word
                    selected_set.remove(w)
                    selected_set.add(best_word)
                    improved = True

            # 2-for-0 deletions and 2-for-1 swaps
            if time.time() < time_limit - 5:
                # We try a limited number of pairs to keep it fast
                pairs_checked = 0
                max_pairs = 200
                while pairs_checked < max_pairs and time.time() < time_limit - 5:
                    i_idx = random.randint(0, len(selected) - 1)
                    j_idx = random.randint(0, len(selected) - 1)
                    if i_idx == j_idx:
                        continue
                    if i_idx > j_idx:
                        i_idx, j_idx = j_idx, i_idx
                    
                    w1 = selected[i_idx]
                    w2 = selected[j_idx]
                    m1 = masks[w1]
                    m2 = masks[w2]

                    # Calculate unique points for w1 and w2 combined
                    # A point is "unique to {w1, w2}" if it is covered by w1 or w2 
                    # and NOT covered by any other word in selected.
                    # counts[pos] is the number of words covering pos.
                    # If counts[pos] == 1, it's covered by exactly one word.
                    # If that word is w1 or w2, it's unique to the pair?
                    # No, if counts[pos] == 1, it's covered by ONLY one word in the whole set.
                    # So if w1 covers pos, and counts[pos]==1, then w2 does NOT cover pos.
                    # So unique points of w1 are those pos in m1 where counts[pos]==1.
                    # Unique points of w2 are those pos in m2 where counts[pos]==1.
                    # The set of points that become uncovered if we remove w1 and w2 is 
                    # U = (m1 & U1) | (m2 & U2) where U1 is unique to w1, U2 unique to w2.
                    # Actually, simpler:
                    # Point p is lost if p is in (m1 | m2) AND counts[p] <= 2 AND 
                    # (p is in m1 OR p is in m2) AND (if counts[p]==2, it must be covered by w1 and w2? No.)
                    
                    # Let's compute the mask of points that are covered ONLY by w1 or ONLY by w2.
                    # Let U1 be mask of points in m1 with counts==1.
                    # Let U2 be mask of points in m2 with counts==1.
                    # If we remove w1 and w2, the points that are lost are U1 | U2.
                    # Why? 
                    # If p in U1, it is covered by w1 and no one else. So removing w1 loses p.
                    # If p in U2, it is covered by w2 and no one else. So removing w2 loses p.
                    # If p is covered by w1 and w2, and no one else, counts[p]=2.
                    # Is p in U1? No, counts[p]=2 != 1.
                    # Is p in U2? No.
                    # But if we remove w1 and w2, p is lost!
                    # So we need to include points where counts[p]==2 and p is in (m1 & m2).
                    
                    u1 = 0
                    mm = m1
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        if counts[pos] == 1:
                            u1 |= lowbit
                        mm ^= lowbit

                    u2 = 0
                    mm = m2
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        if counts[pos] == 1:
                            u2 |= lowbit
                        mm ^= lowbit

                    # Points covered by both w1 and w2 and no one else
                    # These have counts[pos] == 2 and are in m1 & m2
                    u12 = 0
                    mm = m1 & m2
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        if counts[pos] == 2:
                            u12 |= lowbit
                        mm ^= lowbit

                    lost_mask = u1 | u2 | u12

                    if lost_mask == 0:
                        # 2-for-0 deletion
                        mm = m1
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit
                        mm = m2
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit
                        
                        # Remove from selected (handle indices carefully)
                        if i_idx > j_idx:
                            i_idx, j_idx = j_idx, i_idx
                        selected.pop(j_idx)
                        selected.pop(i_idx)
                        selected_set.remove(w1)
                        selected_set.remove(w2)
                        improved = True
                        pairs_checked = max_pairs # Break out
                        break

                    # 2-for-1 swap
                    # Find w3 such that masks[w3] covers lost_mask
                    if lost_mask.bit_count() > 10: # Heuristic: skip if too many unique points
                        pairs_checked += 1
                        continue
                        
                    best_word = -1
                    for w3 in range(total_space):
                        if w3 in selected_set:
                            continue
                        m3 = masks[w3]
                        if (m3 & lost_mask) == lost_mask:
                            best_word = w3
                            break

                    if best_word >= 0:
                        # Perform swap
                        mm = m1
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit
                        mm = m2
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit

                        mm = masks[best_word]
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] += 1
                            mm ^= lowbit

                        if i_idx > j_idx:
                            i_idx, j_idx = j_idx, i_idx
                        selected.pop(j_idx)
                        selected.pop(i_idx)
                        selected.append(best_word)

                        selected_set.remove(w1)
                        selected_set.remove(w2)
                        selected_set.add(best_word)
                        improved = True
                        pairs_checked = max_pairs # Break out
                        break

                    pairs_checked += 1

        return selected

    def run_greedy(randomized=False, alpha=1.0):
        """Greedy set cover."""
        covered = 0
        selected = []
        remaining = full_mask
        selected_set = set()

        while remaining and time.time() < deadline - 10:
            best_gain = -1
            best_word = -1

            for w in range(total_space):
                if w in selected_set:
                    continue
                gain = (masks[w] & remaining).bit_count()
                if randomized:
                    gain = gain + random.gauss(0, 1.0) * alpha
                if gain > best_gain:
                    best_gain = gain
                    best_word = w

            if best_word < 0:
                break

            selected.append(best_word)
            selected_set.add(best_word)
            covered |= masks[best_word]
            remaining = full_mask & ~covered

        return selected

    # Strategy 1: Standard greedy + polish + local search
    if time.time() < deadline - 20:
        sol = run_greedy(randomized=False)
        polished = polish(sol[:], deadline - 20)
        ls = local_search(polished[:], deadline - 20)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Strategy 2: Multiple randomized greedy restarts
    for r in range(50):
        if time.time() > deadline - 30:
            break

        sol = run_greedy(randomized=True, alpha=random.uniform(0.1, 0.8))
        if len(sol) < best_size + 10:
            polished = polish(sol[:], deadline - 30)
            ls = local_search(polished[:], deadline - 30)
            if len(ls) < best_size:
                best_size = len(ls)
                best_selected = ls[:]

    # Final polish
    if time.time() < deadline - 5:
        polished = polish(best_selected[:], deadline - 5)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
