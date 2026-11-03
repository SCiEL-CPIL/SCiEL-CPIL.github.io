import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    # Evaluator kills at 150s; leave safety margin
    deadline = start_time + 142.0

    total_space = q ** n

    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Precompute neighbor lists (sparse representation)
    # Each word covers itself + n*(q-1) neighbors
    neighbors = []
    for i in range(total_space):
        nbrs = [i]
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                nbrs.append(i + diff * powers[pos])
        neighbors.append(nbrs)

    # Also build bitmasks for greedy (needed for fast gain computation)
    masks = [0] * total_space
    for i in range(total_space):
        m = 0
        for nb in neighbors[i]:
            m |= (1 << nb)
        masks[i] = m

    # Precompute coverers for fast local search
    # coverers[x] = list of words w such that x is in neighbors[w]
    coverers = [[] for _ in range(total_space)]
    for w in range(total_space):
        for nb in neighbors[w]:
            coverers[nb].append(w)

    full_mask = (1 << total_space) - 1

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
        Deep local search:
        1. Polish (remove redundant)
        2. 1-out, 1-in swaps
        3. 2-out, 1-in merges
        """
        selected = selected[:]
        if not selected:
            return selected

        # Helper to update counts
        def update_counts(selected, counts):
            counts[:] = 0
            for w in selected:
                mm = masks[w]
                while mm:
                    lowbit = mm & -mm
                    idx = lowbit.bit_length() - 1
                    counts[idx] += 1
                    mm ^= lowbit

        counts = np.zeros(total_space, dtype=np.int32)
        update_counts(selected, counts)

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Polish
            new_selected = polish(selected, time.time() + 1.0) # short polish
            if len(new_selected) < len(selected):
                selected = new_selected
                update_counts(selected, counts)
                improved = True
                continue

            # 2. 1-out, 1-in swaps
            # Try to replace w with w2
            # w2 must cover unique(w) and gain new coverage
            swapped = False
            indices = list(range(len(selected)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > time_limit: break
                w = selected[idx]
                m = masks[w]
                
                # Find unique words covered by w
                unique_words = []
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        unique_words.append(pos)
                    mm ^= lowbit
                
                if not unique_words:
                    # Can remove w
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    selected.pop(idx)
                    swapped = True
                    improved = True
                    break

                # Find candidates covering unique_words
                candidates = set(coverers[unique_words[0]])
                for uw in unique_words[1:]:
                    candidates.intersection_update(coverers[uw])
                
                if not candidates:
                    continue
                
                candidates.discard(w)
                
                best_gain = -1
                best_word = -1
                # We want to maximize new coverage, but must cover unique_words
                # Gain = (m2 & ~covered_by_others).bit_count()
                # covered_by_others = full_mask & ~(m) ... wait, counts tell us coverage
                # Actually, we just want to maximize (m2 & ~m).bit_count() ? No.
                # We want to maximize the number of NEW positions covered by m2 that were not covered by others.
                # Positions covered by others = full_mask & ~ (positions with count==1 and in m) ? No.
                # Let's just compute gain as (m2 & ~m).bit_count() is a good proxy for "newness" relative to w.
                # But we need to ensure the rest of the set still covers everything.
                # Since w2 covers unique_words, and others cover the rest (count > 1 for rest),
                # the union will cover everything if w2 covers unique_words.
                # So we just want to maximize (m2 & ~m).bit_count() to prepare for future merges?
                # Or just take the first one?
                # Let's try to maximize (m2 & ~m).bit_count() to increase "slack".
                
                for w2 in candidates:
                    m2 = masks[w2]
                    gain = (m2 & ~m).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_word = w2

                if best_word >= 0:
                    # Swap
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
                    swapped = True
                    improved = True
                    break

            if swapped:
                continue

            # 3. 2-out, 1-in merges
            # Try to replace w1, w2 with w3
            # w3 must cover unique(w1, w2)
            # unique(w1, w2) = (m1 & m2) positions with count 1? No.
            # A position is unique to {w1, w2} if count[pos] == 2 and (pos in m1 or pos in m2).
            # Actually, if count[pos] == 2, it is covered by exactly 2 words.
            # If we remove w1 and w2, positions with count 1 are lost.
            # Positions with count 2 are covered by one other word + w1 or w2.
            # If a position is in m1 and m2, and count==2, removing both loses it.
            # If a position is in m1 only, and count==2, removing w1 loses it? No, covered by w1 and one other. Removing w1 leaves count 1. So it's safe.
            # Wait.
            # Let C be the set.
            # Remove w1, w2.
            # A position p is uncovered if count[p] == 1 and p in m1 (or m2).
            # Or count[p] == 2 and p in m1 and p in m2? No, if p in m1 and m2, count is at least 2.
            # If count[p] == 2, and p in m1, p in m2, then removing both makes count 0. Uncovered.
            # If count[p] == 2, and p in m1, not in m2, then removing w1 makes count 1. Covered by the other word. Safe.
            # So, the set of positions that MUST be covered by the new word w3 is:
            # U = { p | (count[p] == 1 and p in m1) or (count[p] == 1 and p in m2) or (count[p] == 2 and p in m1 and p in m2) }
            # Actually, simpler:
            # Let M_others = OR of masks of C \ {w1, w2}.
            # Let M_removed = m1 | m2.
            # Uncovered = M_removed & ~M_others.
            # This is exactly the set of positions that are covered ONLY by w1 or w2.
            
            merged = False
            # Sample pairs to save time
            num_pairs = min(1000, len(selected) * (len(selected)-1) // 2)
            pairs = random.sample([(i, j) for i in range(len(selected)) for j in range(i+1, len(selected))], num_pairs)
            
            for i, j in pairs:
                if time.time() > time_limit: break
                w1 = selected[i]
                w2 = selected[j]
                m1 = masks[w1]
                m2 = masks[w2]
                
                # Calculate M_others
                # M_others = full_mask & ~(m1 & (counts == 1)) ? No.
                # Let's just compute the union of others.
                # This is expensive if done from scratch.
                # But we can use counts.
                # A position p is in M_others if count[p] > (1 if p in m1 else 0) + (1 if p in m2 else 0).
                # This is hard to vectorize easily with bitmasks.
                
                # Alternative:
                # Uncovered = (m1 | m2) & ~ (OR of others)
                # OR of others = full_mask & ~ (positions where count <= (1 if p in m1 else 0) + (1 if p in m2 else 0))
                
                # Let's try a different approach for 2-out 1-in.
                # Just try to find w3 that covers unique words of w1 and w2.
                # unique_w1 = {p | count[p]==1 and p in m1}
                # unique_w2 = {p | count[p]==1 and p in m2}
                # shared_unique = {p | count[p]==2 and p in m1 and p in m2}
                # Required = unique_w1 | unique_w2 | shared_unique
                
                req_mask = 0
                mm = m1
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        req_mask |= lowbit
                    elif counts[pos] == 2 and (m2 & lowbit):
                        req_mask |= lowbit
                    mm ^= lowbit
                    
                mm = m2
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        req_mask |= lowbit
                    # shared_unique already handled if we iterate m1 first? 
                    # If p in m1 and m2, and count==2, it was added in m1 loop.
                    # If p in m2 only, and count==1, added here.
                    mm ^= lowbit

                if req_mask == 0:
                    # Can remove both
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
                    # Remove from selected (careful with indices)
                    if i < j:
                        selected.pop(j)
                        selected.pop(i)
                    else:
                        selected.pop(i)
                        selected.pop(j)
                    merged = True
                    improved = True
                    break

                # Find w3 covering req_mask
                candidates = set(coverers[req_mask.bit_length() - 1]) # Just get first bit
                # This is wrong. req_mask is a set of bits.
                # We need intersection of coverers for all bits in req_mask.
                
                bits = []
                mm = req_mask
                while mm:
                    lowbit = mm & -mm
                    bits.append(lowbit.bit_length() - 1)
                    mm ^= lowbit
                
                if not bits:
                    continue
                    
                candidates = set(coverers[bits[0]])
                for b in bits[1:]:
                    candidates.intersection_update(coverers[b])
                
                if not candidates:
                    continue
                
                # Exclude w1, w2
                candidates.discard(w1)
                candidates.discard(w2)
                
                if candidates:
                    w3 = random.choice(list(candidates))
                    # Swap
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
                    
                    mm = masks[w3]
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] += 1
                        mm ^= lowbit
                    
                    if i < j:
                        selected.pop(j)
                        selected.pop(i)
                    else:
                        selected.pop(i)
                        selected.pop(j)
                    selected.append(w3)
                    merged = True
                    improved = True
                    break

        return selected

    def run_greedy():
        covered = 0
        selected = []
        remaining = full_mask

        while remaining and time.time() < deadline - 5:
            best_gain = -1
            best_word = -1
            for w in range(total_space):
                gain = (masks[w] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_word = w

            if best_word < 0:
                break

            selected.append(best_word)
            covered |= masks[best_word]
            remaining = full_mask & ~covered

        return selected

    best_size = total_space
    best_selected = list(range(total_space))

    # Strategy 1: Greedy + Deep Local Search
    if time.time() < deadline - 20:
        sol = run_greedy()
        polished = polish(sol[:], deadline - 20)
        ls = local_search(polished[:], deadline - 20)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Strategy 2: Iterated Local Search
    current = best_selected[:]
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        
        # Perturb
        perturbed = current[:]
        if len(perturbed) > 2:
            # Remove 2, add 2
            idx1, idx2 = random.sample(range(len(perturbed)), 2)
            w1 = perturbed.pop(idx1)
            w2 = perturbed.pop(idx2)
            
            # Add random
            w3 = random.randint(0, total_space - 1)
            w4 = random.randint(0, total_space - 1)
            if w3 not in perturbed:
                perturbed.append(w3)
            if w4 not in perturbed and w4 != w3:
                perturbed.append(w4)
        
        polished = polish(perturbed[:], deadline - 10)
        ls = local_search(polished[:], deadline - 10)
        
        if len(ls) < len(current):
            current = ls[:]
            if len(current) < best_size:
                best_size = len(current)
                best_selected = current[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
