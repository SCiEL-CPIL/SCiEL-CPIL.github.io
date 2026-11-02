import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    # Evaluator kills at 150s. We aim to return well before.
    # Use a safe deadline to ensure we don't get killed.
    deadline = start_time + 140.0

    total_space = q ** n

    # For the specific task K_3(7,1), total_space = 2187.
    # We use Python integers as bitmasks.
    # 2187 bits is ~273 bytes. Operations are fast.

    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i - 1] * q

    # Precompute neighbor bitmasks for all words.
    # mask[i] has bits set for all words within Hamming distance `radius` of i.
    # For radius=1, this is i plus all words differing in exactly 1 position.
    masks = [0] * total_space
    for i in range(total_space):
        m = 1 << i
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                nb = i + diff * powers[pos]
                m |= (1 << nb)
        masks[i] = m

    full_mask = (1 << total_space) - 1

    # Helper: check if a list of indices covers the space
    def is_valid(selected):
        covered = 0
        for w in selected:
            covered |= masks[w]
        return covered == full_mask

    # Helper: get counts of coverage
    def get_counts(selected):
        counts = [0] * total_space
        for w in selected:
            m = masks[w]
            # We need to increment counts for bits set in m
            # This is slow if done naively.
            # But total_space is small (2187).
            # We can iterate over bits.
            # Or use numpy?
            # Let's stick to Python for now, but optimize if needed.
            # Actually, for local search, we need fast count updates.
            # We will maintain counts incrementally in the local search loop.
            # For initial check, we just need validity.
            pass
        return counts

    # Helper: remove redundant words
    def remove_redundant(selected):
        if not selected:
            return selected
        
        # Build counts
        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            # Add mask bits to counts
            # This is O(total_space) per word if we iterate bits.
            # total_space * len(selected) ~ 2187 * 200 ~ 400k ops. Fast enough.
            m = masks[w]
            # Iterate set bits
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            # Shuffle to avoid bias
            indices = list(range(len(selected)))
            random.shuffle(indices)
            
            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                m = masks[w]
                removable = True
                # Check if any bit in m has count == 1
                temp_m = m
                while temp_m:
                    lsb = temp_m & -temp_m
                    idx_bit = lsb.bit_length() - 1
                    if counts[idx_bit] == 1:
                        removable = False
                        break
                    temp_m ^= lsb
                
                if removable:
                    # Remove w
                    m = masks[w]
                    temp_m = m
                    while temp_m:
                        lsb = temp_m & -temp_m
                        idx_bit = lsb.bit_length() - 1
                        counts[idx_bit] -= 1
                        temp_m ^= lsb
                    selected.pop(idx)
                    changed = True
        
        return selected

    # Helper: local search swaps
    def local_search(selected):
        if not selected:
            return selected
        
        # Build counts
        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            m = masks[w]
            temp_m = m
            while temp_m:
                lsb = temp_m & -temp_m
                idx_bit = lsb.bit_length() - 1
                counts[idx_bit] += 1
                temp_m ^= lsb
        
        selected_set = set(selected)
        
        max_iterations = 500
        for _ in range(max_iterations):
            if time.time() > deadline - 2:
                break
            
            # Pick a random word from selected
            idx = random.randint(0, len(selected) - 1)
            w = selected[idx]
            
            # Find unique neighbors (count == 1)
            uniques = []
            m = masks[w]
            temp_m = m
            while temp_m:
                lsb = temp_m & -temp_m
                idx_bit = lsb.bit_length() - 1
                if counts[idx_bit] == 1:
                    uniques.append(idx_bit)
                temp_m ^= lsb
            
            if not uniques:
                # Can remove w
                m = masks[w]
                temp_m = m
                while temp_m:
                    lsb = temp_m & -temp_m
                    idx_bit = lsb.bit_length() - 1
                    counts[idx_bit] -= 1
                    temp_m ^= lsb
                selected.pop(idx)
                selected_set.discard(w)
                continue
            
            # Try to find a replacement w'
            # w' must cover all uniques.
            # Prefer w' in selected (removal) or w' not in selected (swap).
            # To be efficient, we can check candidates that cover at least one unique.
            
            # Simple strategy: scan all words in space? 2187 is small.
            # But we can be smarter.
            # Let's just scan all words for now, as 2187 is small.
            
            best_candidate = None
            best_score = -1
            is_removal = False
            
            # Check if any existing word covers all uniques (except w itself)
            # If so, we can just remove w.
            # This is handled by the "unique" check above? No, that checks if w is redundant.
            # Here we want to swap w with w'.
            
            # We want to minimize the "cost" of w'.
            # Cost = number of new words covered by w' that were not covered by w?
            # Or simply: we just need validity.
            
            # Let's try to find any w' that covers all uniques.
            # If w' is in selected, we can remove w and keep w' (size -1).
            # If w' is not in selected, we swap w with w' (size 0 change).
            
            # To make progress, we should prefer swaps that allow future removals?
            # For now, just find any valid swap.
            
            for w2 in range(total_space):
                if w2 == w:
                    continue
                
                # Check if w2 covers all uniques
                m2 = masks[w2]
                ok = True
                for u in uniques:
                    if not (m2 >> u) & 1:
                        ok = False
                        break
                
                if not ok:
                    continue
                
                # If w2 is in selected, it's a removal candidate (size -1)
                # But wait, if w2 is in selected, and w2 covers uniques of w,
                # then removing w is valid.
                # We already checked if w is redundant (all counts > 1).
                # If w is not redundant, but some other word w2 covers the uniques,
                # does that mean w is redundant?
                # No. The uniques are covered ONLY by w.
                # If w2 covers them, then after adding w2 (or if w2 is already there),
                # the count for those uniques becomes > 1.
                # So yes, if w2 is in selected and covers uniques, w is redundant.
                # But we already tried to remove redundant words.
                # So if we are here, w is NOT redundant.
                # This implies no single other word in selected covers ALL uniques.
                # Or maybe my redundancy check was incomplete?
                # Redundancy check: for all nb in neighbors[w], counts[nb] > 1.
                # This is equivalent to: no nb has count == 1.
                # So if uniques is not empty, w is not redundant.
                # Therefore, no w2 in selected covers all uniques.
                # So we must look for w2 NOT in selected.
                
                if w2 in selected_set:
                    continue
                
                # Found a candidate w2 not in selected that covers uniques.
                # Perform swap: remove w, add w2.
                # Update counts.
                
                # Remove w
                m = masks[w]
                temp_m = m
                while temp_m:
                    lsb = temp_m & -temp_m
                    idx_bit = lsb.bit_length() - 1
                    counts[idx_bit] -= 1
                    temp_m ^= lsb
                
                # Add w2
                m2 = masks[w2]
                temp_m = m2
                while temp_m:
                    lsb = temp_m & -temp_m
                    idx_bit = lsb.bit_length() - 1
                    counts[idx_bit] += 1
                    temp_m ^= lsb
                
                selected[idx] = w2
                selected_set.discard(w)
                selected_set.add(w2)
                
                best_candidate = w2
                break # Found one, break. We can try to be smarter later.
            
            if best_candidate is None:
                # No swap found for this w.
                # Try another random w?
                # Or stop?
                # Let's just continue to next iteration.
                pass
        
        return selected

    # Greedy construction
    def run_greedy():
        covered = 0
        selected = []
        remaining = full_mask
        
        while remaining:
            best_gain = -1
            best_word = -1
            
            for w in range(total_space):
                gain = (masks[w] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            
            if best_word == -1:
                break
            
            selected.append(best_word)
            covered |= masks[best_word]
            remaining = full_mask & ~covered
            
            if time.time() > deadline - 10:
                break
        
        return selected

    def run_greedy_randomized():
        covered = 0
        selected = []
        remaining = full_mask
        
        while remaining and time.time() < deadline - 5:
            max_gain = -1
            candidates = []
            
            for w in range(total_space):
                gain = (masks[w] & remaining).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    candidates = [w]
                elif gain == max_gain:
                    candidates.append(w)
            
            if not candidates:
                break
                
            best_word = random.choice(candidates)
            
            selected.append(best_word)
            covered |= masks[best_word]
            remaining = full_mask & ~covered
            
        return selected

    best_size = total_space
    best_selected = list(range(total_space))
    
    # 1. Standard Greedy
    if time.time() < deadline - 20:
        sol = run_greedy()
        if len(sol) < best_size:
            best_size = len(sol)
            best_selected = sol[:]
            
            # Polish
            if time.time() < deadline - 10:
                polished = remove_redundant(best_selected[:])
                polished = local_search(polished)
                if len(polished) < best_size:
                    best_size = len(polished)
                    best_selected = polished[:]

    # 2. Randomized Greedy Restarts
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        sol = run_greedy_randomized()
        if len(sol) < best_size:
            best_size = len(sol)
            best_selected = sol[:]
            
            # Polish
            if time.time() < deadline - 5:
                polished = remove_redundant(best_selected[:])
                polished = local_search(polished)
                if len(polished) < best_size:
                    best_size = len(polished)
                    best_selected = polished[:]
        
        # If we've done enough restarts, stop
        if restarts > 100:
            break

    # Final polish on the best found
    if time.time() < deadline - 2:
        polished = remove_redundant(best_selected[:])
        polished = local_search(polished)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    # Decode
    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    best_words = [int_to_tuple(v) for v in best_selected]
    
    return best_words
