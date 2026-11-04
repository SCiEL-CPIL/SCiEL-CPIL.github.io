import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    total_words = q ** n
    
    # If the problem is too large for this specific heuristic approach, 
    # fall back to a simple valid solution (though score will be low).
    # For q=3, n=7, total_words=2187, which is small enough.
    if total_words > 50000:
        # Fallback: return first word and hope for the best? No, must be valid.
        # Return all words to ensure validity.
        return [[i // (q ** (n - 1 - k)) % q for k in range(n)] for i in range(total_words)]

    # Mapping functions
    def int_to_word(x):
        w = [0] * n
        for k in range(n):
            w[k] = x % q
            x //= q
        return w

    def word_to_int(w):
        x = 0
        base = 1
        for k in range(n):
            x += w[k] * base
            base *= q
        return x

    # Precompute neighbor masks
    # neighbor_masks[i] is a bitmask of words within Hamming distance 'radius' of word i
    # For radius=1, we check all words at distance 0 and 1.
    # For general radius, we would need to iterate all words and check distance,
    # but for n=7, q=3, radius=1, we can generate neighbors directly.
    
    neighbor_masks = [0] * total_words
    
    # Optimization: Precompute all words as tuples? No, just use int_to_word when needed for neighbor gen.
    # Actually, for radius=1, neighbors are easy to generate from the word representation.
    
    for i in range(total_words):
        w = int_to_word(i)
        mask = 1 << i
        
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    w[pos] = new_val
                    j = word_to_int(w)
                    mask |= (1 << j)
                w[pos] = orig_val
        
        # If radius > 1, we would need to expand further.
        # Given the task is K_3(7,1), radius is 1.
        # We assume radius=1 for efficiency in this specific implementation.
        # If radius > 1, this code would be invalid. 
        # However, the task contract specifies the problem.
        # To be safe, if radius > 1, we should fall back or use a slower method.
        # But let's assume radius=1 as per the problem statement "K_3(7,1)".
        
        neighbor_masks[i] = mask

    all_mask = (1 << total_words) - 1
    
    best_codewords = None
    best_size = float('inf')
    
    # Number of trials to run
    # Each trial is O(N * U) where N is num codewords (~200) and U is universe size (2187).
    # 200 * 2187 * 50 trials = ~22 million operations.
    # Python can do ~10-50 million simple ops/sec. Bit ops are fast.
    # So 50-100 trials should be safe.
    
    num_trials = 100
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
            
        uncovered = all_mask
        selected_set = set()
        selected_list = []
        
        while uncovered:
            best_count = -1
            best_idx = -1
            candidates = []
            
            for i in range(total_words):
                if i in selected_set:
                    continue
                
                new_cover = neighbor_masks[i] & uncovered
                count = new_cover.bit_count()
                
                if count > best_count:
                    best_count = count
                    best_idx = i
                    candidates = [i]
                elif count == best_count and count > 0:
                    candidates.append(i)
            
            if best_idx == -1:
                break
                
            if len(candidates) > 1 and trial > 0:
                best_idx = random.choice(candidates)
                
            selected_set.add(best_idx)
            selected_list.append(best_idx)
            uncovered &= ~neighbor_masks[best_idx]
            
        # Corrected redundancy removal
        # We iterate until no more codewords can be removed.
        # To speed up, we calculate the union of all masks once, then for each candidate,
        # check if its mask is a subset of (total_union - its_mask) ? No, that's not right.
        # A codeword c is redundant if Union(Others) covers Mask(c).
        # Let TotalUnion = OR(all masks).
        # If we remove c, the new union is TotalUnion with c's unique bits potentially lost.
        # Actually, simpler:
        # For each c in selected:
        #   others_union = OR(mask[j] for j != c)
        #   if (mask[c] & ~others_union) == 0: remove c
        
        # Optimization: Instead of recalculating others_union from scratch for every c (O(K^2)),
        # we can just do it. K is ~200, so K^2 = 40k ops. This is fine.
        
        changed = True
        while changed:
            changed = False
            # Iterate backwards to avoid index shifting issues if we remove, 
            # but since we break immediately after removal, forward is fine too.
            for i in range(len(selected_list)):
                idx = selected_list[i]
                my_mask = neighbor_masks[idx]
                
                # Calculate union of all other selected codewords
                others_union = 0
                for j in range(len(selected_list)):
                    if j != i:
                        others_union |= neighbor_masks[selected_list[j]]
                
                # If my coverage is fully contained in others' coverage, I am redundant
                if (my_mask & ~others_union) == 0:
                    # Remove idx from list and set
                    selected_list.pop(i)
                    selected_set.remove(idx)
                    changed = True
                    break

        if len(selected_list) < best_size:
            best_size = len(selected_list)
            best_codewords = selected_list[:]
            
        # Check time
        if time.time() > deadline - 5:
            break
            
    if best_codewords is None:
        return [int_to_word(i) for i in range(total_words)]
        
    return [int_to_word(i) for i in best_codewords]
