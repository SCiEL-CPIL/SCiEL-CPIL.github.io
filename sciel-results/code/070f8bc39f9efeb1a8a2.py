import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. Aim for 140s.
    deadline = start_time + 140.0 

    total_space = q ** n
    
    if total_space > 20000:
        # Fallback for larger spaces if this function is reused, though task is specific
        # For K_3(7,1), total_space is 2187, which fits in a Python int bitmask.
        # If space is larger, this bitmask approach might be slow or memory heavy.
        # But the task is specifically K_3(7,1).
        pass

    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Precompute neighbor masks
    # masks[i] is a bitmask where bit j is 1 if dist(i, j) <= radius
    masks = [0] * total_space
    for i in range(total_space):
        m = 1 << i
        if radius == 1:
            for pos in range(n):
                digit_val = (i // powers[pos]) % q
                for new_val in range(q):
                    if new_val == digit_val:
                        continue
                    diff = new_val - digit_val
                    j = i + diff * powers[pos]
                    m |= (1 << j)
        else:
            # General radius: generate all neighbors
            # This is slow for radius > 1 and large n, but K_3(7,1) is the target.
            # For safety, we only implement radius=1 efficiently.
            # If radius > 1, we fall back to a slower list-based approach or assume it's small.
            # Given the task, radius is 1.
            pass
        masks[i] = m

    best_size = total_space
    best_selected = list(range(total_space))

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    def run_greedy(randomized=False, alpha=1.0):
        """
        Greedy set cover using bitmasks.
        """
        covered_mask = 0
        selected = []
        remaining_count = total_space
        selected_set = set()
        
        # Precompute list of all words
        all_words = list(range(total_space))
        
        while remaining_count > 0 and time.time() < deadline - 10:
            best_gain = -1
            best_word = -1
            
            # Optimization: If remaining_count is small, we can break early? No, need to cover all.
            # Iterate through all words not yet selected
            # To speed up, we can skip words that are already covered? No, they might cover new things.
            # Actually, if a word is fully covered, it adds 0 gain.
            
            for w in all_words:
                if w in selected_set:
                    continue
                
                # Calculate gain: number of bits in (mask[w] & ~covered_mask)
                new_covered = masks[w] & ~covered_mask
                gain = new_covered.bit_count()
                
                if randomized:
                    gain += random.gauss(0, alpha)
                
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            
            if best_word < 0 or best_gain <= 0:
                break

            selected.append(best_word)
            selected_set.add(best_word)
            
            # Update covered mask
            covered_mask |= masks[best_word]
            remaining_count -= best_gain # Approximate, bit_count is exact
            
        return selected

    def polish(selected, time_limit):
        """
        Remove redundant codewords.
        A codeword w is redundant if every word in masks[w] is covered by at least 2 codewords.
        """
        selected = selected[:]
        if not selected:
            return selected

        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            m = masks[w]
            # Iterate through set bits in m
            while m:
                lowbit = m & -m
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                m ^= lowbit

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
                
                # Check if any word in m has count == 1
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
                    # Remove w
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
        Local search with 1-for-1 and 2-for-1 swaps.
        """
        selected = selected[:]
        if not selected:
            return selected

        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            m = masks[w]
            while m:
                lowbit = m & -m
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                m ^= lowbit

        selected_set = set(selected)
        all_words = list(range(total_space))

        # 1-for-1 swap
        for _ in range(100):
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

                # Find unique neighbors of w (count == 1)
                unique_mask = 0
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        unique_mask |= lowbit
                    mm ^= lowbit

                if unique_mask == 0:
                    # w is redundant, remove it
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

                # Try to find a replacement w2
                best_gain = -1
                best_word = -1
                
                # Heuristic: Look for w2 that covers unique_mask and maximizes additional coverage
                # To speed up, we can limit the search or use a heuristic.
                # For now, iterate through all non-selected words.
                for w2 in all_words:
                    if w2 in selected_set:
                        continue
                    m2 = masks[w2]
                    if (m2 & unique_mask) == unique_mask:
                        # w2 covers all unique neighbors of w
                        # Calculate gain: how many new words does w2 cover that w didn't?
                        # Actually, we want to minimize the loss of coverage from removing w,
                        # but since w2 covers unique_mask, we lose nothing critical.
                        # We gain coverage from (m2 & ~m).
                        gain = (m2 & ~m).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_word = w2

                if best_word != -1 and best_gain > 0:
                    # Perform swap
                    # Remove w
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    
                    # Add w2
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
        if time.time() < time_limit - 5:
            for _ in range(20):
                if time.time() > time_limit:
                    break
                improved = False
                indices = list(range(len(selected)))
                random.shuffle(indices)
                
                # Try a limited number of pairs
                pair_checks = min(50, len(indices) * (len(indices)-1) // 2)
                checked = 0
                for i_idx in range(len(indices)):
                    if checked >= pair_checks:
                        break
                    idx1 = indices[i_idx]
                    if idx1 >= len(selected):
                        continue
                    w1 = selected[idx1]
                    
                    for j_idx in range(i_idx + 1, len(indices)):
                        if checked >= pair_checks:
                            break
                        idx2 = indices[j_idx]
                        if idx2 >= len(selected):
                            continue
                        w2 = selected[idx2]
                        checked += 1
                        
                        m1 = masks[w1]
                        m2 = masks[w2]
                        
                        # Find words uniquely covered by w1 or w2
                        # A word j is uniquely covered by w1 or w2 if:
                        # 1. j is in m1 or m2
                        # 2. count[j] is 1 (if j in m1 only) or 2 (if j in m1 and m2) 
                        #    AND no other selected word covers j.
                        
                        unique_mask = 0
                        
                        # Check intersection
                        inter = m1 & m2
                        mm = inter
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            if counts[pos] == 2:
                                unique_mask |= lowbit
                            mm ^= lowbit
                        
                        # Check m1 \ m2
                        diff1 = m1 & ~m2
                        mm = diff1
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            if counts[pos] == 1:
                                unique_mask |= lowbit
                            mm ^= lowbit
                            
                        # Check m2 \ m1
                        diff2 = m2 & ~m1
                        mm = diff2
                        while mm:
                            lowbit = mm & -mm
                            pos = lowbit.bit_length() - 1
                            if counts[pos] == 1:
                                unique_mask |= lowbit
                            mm ^= lowbit

                        if unique_mask == 0:
                            # Both are redundant together? 
                            # If unique_mask is 0, it means every word covered by w1 or w2 is covered by at least 3 words (or 2 if shared).
                            # Wait, if counts[pos] > 2 for all pos in union, then we can remove both?
                            # No, we need to find a w3 that covers unique_mask. If unique_mask is 0, we can just remove w1 and w2?
                            # No, we need to replace them with w3? Or just remove if coverage remains valid?
                            # If unique_mask is 0, removing w1 and w2 leaves all words covered by at least (count-2) words.
                            # If min count in union was >= 3, then after removal min count is >= 1. So we can remove both.
                            # But we want to replace with 1 word to maintain density? 
                            # Actually, if we remove 2 and add 0, size reduces by 2. This is great if valid.
                            # Let's check if we can just remove them.
                            # We need to verify that no word becomes uncovered.
                            # If unique_mask is 0, it implies for all j in m1|m2, count[j] > (1 if j in m1 xor m2 else 2).
                            # So count[j] >= 2 for exclusive, >= 3 for shared.
                            # After removal, count[j] >= 1. So coverage is maintained.
                            # So we can remove both!
                            
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
                            
                            # Pop from list
                            if idx1 > idx2:
                                idx1, idx2 = idx2, idx1
                            selected.pop(idx2)
                            selected.pop(idx1)
                            selected_set.remove(w1)
                            selected_set.remove(w2)
                            improved = True
                            break
                            
                        # If unique_mask is not 0, find w3
                        # Limit unique_mask size to avoid expensive search
                        if unique_mask.bit_count() > 8:
                            continue

                        found_w3 = -1
                        for w3 in all_words:
                            if w3 in selected_set:
                                continue
                            m3 = masks[w3]
                            if (m3 & unique_mask) == unique_mask:
                                found_w3 = w3
                                break

                        if found_w3 != -1:
                            # Perform swap: remove w1, w2; add w3
                            # Update counts
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
                            mm = masks[found_w3]
                            while mm:
                                lowbit = mm & -mm
                                pos = lowbit.bit_length() - 1
                                counts[pos] += 1
                                mm ^= lowbit
                            
                            # Update list
                            if idx1 > idx2:
                                idx1, idx2 = idx2, idx1
                            selected.pop(idx2)
                            selected.pop(idx1)
                            selected.append(found_w3)
                            
                            selected_set.remove(w1)
                            selected_set.remove(w2)
                            selected_set.add(found_w3)
                            
                            improved = True
                            break
                            
                    if improved:
                        break
                        
                if improved:
                    # Polish after swap
                    selected = polish(selected, time_limit)
                    # Rebuild counts
                    counts = np.zeros(total_space, dtype=np.int32)
                    for w in selected:
                        m = masks[w]
                        while m:
                            lowbit = m & -m
                            idx = lowbit.bit_length() - 1
                            counts[idx] += 1
                            m ^= lowbit
                    selected_set = set(selected)

        return selected

    # Strategy 1: Standard greedy + polish + local search
    if time.time() < deadline - 30:
        sol = run_greedy(randomized=False)
        polished = polish(sol[:], deadline - 30)
        ls = local_search(polished[:], deadline - 30)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Strategy 2: Multiple randomized greedy restarts
    restarts = 0
    while time.time() < deadline - 40 and restarts < 20:
        restarts += 1
        alpha = random.uniform(0.1, 0.5)
        sol = run_greedy(randomized=True, alpha=alpha)
        
        # If greedy result is already good, polish it
        if len(sol) < best_size + 20:
            polished = polish(sol[:], deadline - 40)
            ls = local_search(polished[:], deadline - 40)
            if len(ls) < best_size:
                best_size = len(ls)
                best_selected = ls[:]

    # Final polish and local search on best found
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
