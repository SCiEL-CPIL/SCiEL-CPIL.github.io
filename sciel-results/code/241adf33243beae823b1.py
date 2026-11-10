import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit

    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return list(itertools.product(range(q), repeat=n))

    total_words = q ** n  # 2187

    # Precompute all words and their indices
    all_words = []
    for i in range(total_words):
        w = [0] * n
        idx = i
        for pos in range(n - 1, -1, -1):
            w[pos] = idx % q
            idx //= q
        all_words.append(w)

    # Precompute spheres as Python integers (bitsets)
    # sphere[i] is a bitmask where bit j is 1 if word j is in sphere of word i
    spheres = [0] * total_words
    # Precompute coverers: coverers[j] is a list of i such that j is in sphere[i]
    coverers = [[] for _ in range(total_words)]
    
    # Helper to compute index from word
    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx

    for i in range(total_words):
        w = all_words[i]
        mask = 1 << i
        spheres[i] = mask
        coverers[i].append(i)
        
        for pos in range(n):
            for delta in [-1, 1]:
                new_val = (w[pos] + delta) % q
                if new_val != w[pos]:
                    nw = w.copy()
                    nw[pos] = new_val
                    j = word_to_idx(nw)
                    spheres[i] |= (1 << j)
                    coverers[j].append(i)

    # Convert coverers to sets for faster membership? No, lists are fine for iteration.
    # But for "find c_new covering E", we need to intersect coverers of words in E.
    
    def popcount(x):
        return bin(x).count('1')

    def greedy_cover(uncovered_mask):
        code = set()
        unc = uncovered_mask
        while unc:
            best_gain = -1
            best_c = -1
            # We can't iterate all 2187 every time if it's slow, but 2187 is small.
            # Optimization: Only consider candidates that cover at least one uncovered word.
            # But checking all is O(2187) which is fine.
            for c in range(total_words):
                if c in code:
                    continue
                gain = popcount(spheres[c] & unc)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                    if gain == 15: # Max possible
                        break
            if best_c == -1:
                break
            code.add(best_c)
            unc &= ~spheres[best_c]
        return code

    def prune_code(code):
        # Returns pruned code
        # We maintain a count array to check redundancy quickly
        # But since code size is small (~170), and sphere size is 15, 
        # we can just recalculate counts or use a dict.
        # Let's use a dictionary for counts of words in the current code's union.
        # Actually, simple list of counts is faster.
        counts = [0] * total_words
        for c in code:
            # Iterate bits in spheres[c]
            mask = spheres[c]
            while mask:
                # Get lowest set bit
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                mask ^= lsb
        
        new_code = set()
        for c in code:
            mask = spheres[c]
            min_count = 15 # Max is 15
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if counts[idx] < min_count:
                    min_count = counts[idx]
                m ^= lsb
            
            if min_count > 1:
                # Redundant, do not add
                pass
            else:
                new_code.add(c)
                # If we keep c, we don't update counts here, 
                # but for correctness of subsequent checks in this pass, 
                # we'd need to know if c was removed.
                # However, removing a codeword only DECREASES counts.
                # If min_count > 1, it means every word in S(c) is covered by >= 2 codewords in `code`.
                # If we remove c, counts drop by 1.
                # If we keep c, counts stay.
                # The standard "greedy pruning" is: remove if redundant.
                # This is safe to do in one pass if we are careful?
                # Actually, if A is redundant because B covers it, and B is redundant because A covers it, 
                # we might remove both. 
                # To be safe, we can just iterate and remove if min_count > 1 based on CURRENT code.
                # But if we remove A, B might become non-redundant.
                # So we should do a loop.
        
        # Better: Iterative removal
        current_code = set(code)
        changed = True
        while changed:
            changed = False
            # Recalculate counts for current_code
            counts = [0] * total_words
            for c in current_code:
                mask = spheres[c]
                while mask:
                    lsb = mask & -mask
                    idx = lsb.bit_length() - 1
                    counts[idx] += 1
                    mask ^= lsb
            
            to_remove = []
            for c in current_code:
                mask = spheres[c]
                min_count = 15
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] < min_count:
                        min_count = counts[idx]
                    m ^= lsb
                if min_count > 1:
                    to_remove.append(c)
            
            for c in to_remove:
                current_code.remove(c)
                changed = True
                
        return current_code

    def local_search(code):
        code = prune_code(code)
        
        # Calculate counts for swap analysis
        counts = [0] * total_words
        for c in code:
            mask = spheres[c]
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                mask ^= lsb
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            code = prune_code(code)
            
            # Recalculate counts
            counts = [0] * total_words
            for c in code:
                mask = spheres[c]
                while mask:
                    lsb = mask & -mask
                    idx = lsb.bit_length() - 1
                    counts[idx] += 1
                    mask ^= lsb
            
            # Find critical codewords
            criticals = []
            for c in code:
                mask = spheres[c]
                min_count = 15
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] < min_count:
                        min_count = counts[idx]
                    m ^= lsb
                if min_count == 1:
                    criticals.append(c)
            
            random.shuffle(criticals)
            
            for c_old in criticals:
                if time.time() > deadline:
                    break
                if c_old not in code:
                    continue
                
                # Find exclusive words of c_old
                exclusive_words = []
                mask = spheres[c_old]
                while mask:
                    lsb = mask & -mask
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        exclusive_words.append(idx)
                    mask ^= lsb
                
                if not exclusive_words:
                    continue # Should not happen for critical, but safe
                
                # Find c_new that covers all exclusive_words
                # Intersection of coverers[ew] for ew in exclusive_words
                # Start with coverers[exclusive_words[0]]
                candidates = coverers[exclusive_words[0]]
                
                # Filter candidates
                valid_candidates = []
                for c_new in candidates:
                    if c_new in code:
                        continue
                    # Check if c_new covers all other exclusive words
                    covers_all = True
                    for ew in exclusive_words[1:]:
                        if not (spheres[c_new] & (1 << ew)):
                            covers_all = False
                            break
                    if covers_all:
                        valid_candidates.append(c_new)
                
                if valid_candidates:
                    # Pick one randomly
                    c_new = random.choice(valid_candidates)
                    
                    # Perform swap
                    code.remove(c_old)
                    code.add(c_new)
                    improved = True
                    # Break to re-prune and recalculate
                    break
                    
        return code

    best_code = None
    best_size = float('inf')
    
    # Run multiple instances of ILS
    num_runs = 0
    while time.time() < deadline:
        num_runs += 1
        
        # 1. Greedy Initialization
        # Add some randomness to greedy: pick best, or second best, etc.
        # Or just standard greedy.
        initial_code = greedy_cover((1 << total_words) - 1)
        
        # 2. Local Search
        code = local_search(initial_code)
        
        # 3. Perturbation (ILS step)
        # If we have time, perturb and re-search
        if time.time() < deadline - 5: # Need time for next search
            # Remove k random codewords
            k = random.randint(2, 4)
            if len(code) > k:
                remove_list = random.sample(list(code), k)
                for c in remove_list:
                    code.remove(c)
                # Re-cover uncovered words
                # Calculate uncovered
                counts = [0] * total_words
                for c in code:
                    mask = spheres[c]
                    while mask:
                        lsb = mask & -mask
                        idx = lsb.bit_length() - 1
                        counts[idx] += 1
                        mask ^= lsb
                
                uncovered_mask = 0
                for i in range(total_words):
                    if counts[i] == 0:
                        uncovered_mask |= (1 << i)
                        
                # Greedy add to cover uncovered
                # This is a mini-greedy
                unc = uncovered_mask
                while unc and time.time() < deadline - 2:
                    best_gain = -1
                    best_c = -1
                    for c in range(total_words):
                        if c in code:
                            continue
                        gain = popcount(spheres[c] & unc)
                        if gain > best_gain:
                            best_gain = gain
                            best_c = c
                            if gain == 15:
                                break
                    if best_c == -1:
                        break
                    code.add(best_c)
                    unc &= ~spheres[best_c]
                
                # Local Search again
                code = local_search(code)
                
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            if best_size <= 160: # Close to lower bound, stop early?
                break

    if best_code is None:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    result = [all_words[i] for i in best_code]
    return result
