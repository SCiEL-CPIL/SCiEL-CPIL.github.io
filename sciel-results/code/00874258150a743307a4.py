import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for 150s limit

    total_space = q ** n
    
    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Precompute neighbor bitmasks
    # masks[i] is a bitmask of all words within Hamming distance 'radius' of i
    masks = [0] * total_space
    for i in range(total_space):
        m = 1 << i
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                m |= (1 << (i + diff * powers[pos]))
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

        # counts[x] is the number of codewords that cover x
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
                
                # Check if any position in the mask has count == 1
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
        Local search with 1-for-1 swaps and 2-for-1 swaps.
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

        # 1-for-1 swap (and deletion if gain > 0)
        # We iterate multiple times to allow cascading improvements
        max_passes = 50
        for p in range(max_passes):
            if time.time() > time_limit:
                break
            improved = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                m = masks[w]

                # Identify unique points covered by w
                unique_mask = 0
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        unique_mask |= lowbit
                    mm ^= lowbit

                if unique_mask == 0:
                    # w is redundant
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

                # Try to find a replacement w2 that covers unique_mask
                # and adds some new coverage (gain)
                best_gain = -1
                best_word = -1
                # To speed up, we can restrict search to words that intersect with unique_mask
                # But for 2187 words, full scan is okay if done sparsely
                # Let's do a random sample of 100 words first
                candidates = random.sample([w2 for w2 in range(total_space) if w2 not in selected_set], min(100, total_space - len(selected)))
                
                for w2 in candidates:
                    m2 = masks[w2]
                    if (m2 & unique_mask) == unique_mask:
                        # w2 covers all unique points of w
                        gain = (m2 & ~m).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_word = w2

                if best_word >= 0 and best_gain > 0:
                    # Perform swap
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

            if not improved:
                break

        # 2-for-1 swap
        # Try to remove 2 words and add 1
        if time.time() < time_limit - 5:
            for _ in range(10):
                if time.time() > time_limit:
                    break
                improved = False
                indices = list(range(len(selected)))
                random.shuffle(indices)
                
                # Try a random subset of pairs
                num_pairs_to_try = min(50, len(indices) * (len(indices) - 1) // 2)
                
                for _ in range(num_pairs_to_try):
                    i_idx = random.randint(0, len(indices) - 1)
                    j_idx = random.randint(0, len(indices) - 1)
                    if i_idx == j_idx:
                        continue
                    idx1 = indices[i_idx]
                    idx2 = indices[j_idx]
                    if idx1 >= len(selected) or idx2 >= len(selected):
                        continue
                    w1 = selected[idx1]
                    w2 = selected[idx2]
                    
                    m1 = masks[w1]
                    m2 = masks[w2]
                    m12 = m1 | m2
                    
                    # Calculate what is covered by others
                    # Instead of summing all others, we can check if (m12 & ~U_others) is subset of some w3
                    # U_others is expensive to compute.
                    # Heuristic: Check if m12 is "small" or if w1, w2 overlap a lot.
                    
                    # Check unique coverage of w1 and w2 together
                    # A point is in the "gap" if it is in m12 and NOT in any other word.
                    # This is equivalent to: counts[pos] == 1 for pos in m12 \ (m1|m2)? No.
                    # counts[pos] counts all words.
                    # If we remove w1 and w2, the count for pos decreases by 1 if pos in m1, by 1 if pos in m2.
                    # If pos is in both, it decreases by 2.
                    # We need the new count to be >= 1 for all pos in m12.
                    # So for pos in m12:
                    #   if pos in m1 and pos in m2: counts[pos] - 2 >= 1 => counts[pos] >= 3
                    #   if pos in m1 xor m2: counts[pos] - 1 >= 1 => counts[pos] >= 2
                    
                    gap_mask = 0
                    mm = m12
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        in_m1 = (m1 & lowbit) != 0
                        in_m2 = (m2 & lowbit) != 0
                        if in_m1 and in_m2:
                            if counts[pos] < 3:
                                gap_mask |= lowbit
                        else:
                            if counts[pos] < 2:
                                gap_mask |= lowbit
                        mm ^= lowbit
                        
                    if gap_mask == 0:
                        # We can remove w1 and w2 without adding anything?
                        # This is a 2-for-0 move.
                        # Remove w1
                        mm = m1
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit
                        # Remove w2
                        mm = m2
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            counts[pos] -= 1
                            mm ^= lowbit
                        # Update indices
                        if idx1 > idx2:
                            idx1, idx2 = idx2, idx1
                        selected.pop(idx2)
                        selected.pop(idx1)
                        selected_set.remove(w1)
                        selected_set.remove(w2)
                        improved = True
                        break
                    else:
                        # We need a w3 that covers gap_mask
                        # Search for w3 in random sample
                        candidates = random.sample([w3 for w3 in range(total_space) if w3 not in selected_set], min(50, total_space - len(selected)))
                        best_word = -1
                        for w3 in candidates:
                            m3 = masks[w3]
                            if (m3 & gap_mask) == gap_mask:
                                best_word = w3
                                break
                                
                        if best_word >= 0:
                            # Perform 2-for-1
                            # Remove w1
                            mm = m1
                            while mm:
                                lowbit = mm & -mm
                                pos = lowbit.bit_length() - 1
                                counts[pos] -= 1
                                mm ^= lowbit
                            # Remove w2
                            mm = m2
                            while mm:
                                lowbit = mm & -mm
                                pos = lowbit.bit_length() - 1
                                counts[pos] -= 1
                                mm ^= lowbit
                            # Add w3
                            mm = masks[best_word]
                            while mm:
                                lowbit = mm & -mm
                                pos = lowbit.bit_length() - 1
                                counts[pos] += 1
                                mm ^= lowbit
                                
                            if idx1 > idx2:
                                idx1, idx2 = idx2, idx1
                            selected.pop(idx2)
                            selected.pop(idx1)
                            selected.append(best_word)
                            
                            selected_set.remove(w1)
                            selected_set.remove(w2)
                            selected_set.add(best_word)
                            improved = True
                            break
                            
                if improved:
                    # Re-polish
                    selected = polish(selected, time_limit)
                    # Rebuild counts
                    counts = np.zeros(total_space, dtype=np.int32)
                    for w in selected:
                        mm = masks[w]
                        while mm:
                            lowbit = mm & -mm
                            idx = lowbit.bit_length() - 1
                            counts[idx] += 1
                            mm ^= lowbit
                    selected_set = set(selected)

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

            # Randomized selection to avoid deterministic traps
            if randomized:
                # Sample a subset of words to speed up and introduce randomness
                # But we need to find the *best* gain, so sampling might miss the best word.
                # Instead, we add noise to the gain.
                pass

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
    restarts = 0
    while time.time() < deadline - 30 and restarts < 50:
        restarts += 1
        sol = run_greedy(randomized=True, alpha=random.uniform(0.1, 0.5))
        if len(sol) < best_size + 10:
            polished = polish(sol[:], deadline - 30)
            ls = local_search(polished[:], deadline - 30)
            if len(ls) < best_size:
                best_size = len(ls)
                best_selected = ls[:]

    # Final polish and local search on the best solution found
    if time.time() < deadline - 5:
        polished = polish(best_selected[:], deadline - 5)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]
            
    if time.time() < deadline - 2:
        final_ls = local_search(best_selected[:], deadline - 2)
        if len(final_ls) < best_size:
            best_size = len(final_ls)
            best_selected = final_ls[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
