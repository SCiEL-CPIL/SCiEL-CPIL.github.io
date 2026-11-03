import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = 120.0 # 120s limit, keep margin
    
    # Specific to the task: q=3, n=6, radius=1
    # But we should handle general inputs if possible, though the task is specific.
    # The prompt says "evaluator passes this task's alphabet size q, length n and covering radius".
    # So we assume q=3, n=6, radius=1 but should code defensively or specifically.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback to seed-like behavior or generic greedy
        # For now, implement generic greedy for small sizes, but optimize for 3,6,1
        pass

    total_words = q ** n
    # For 3, 6: 729 words.
    
    # Generate all words as tuples of ints
    # And map to index 0..728
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute coverage masks
    # mask[i] is a Python int representing the set of indices covered by words[i]
    masks = [0] * total_words
    
    for i, w in enumerate(words):
        m = 1 << i # covers itself
        # radius 1: change one position
        for pos in range(n):
            for val in range(q):
                if val == w[pos]: continue
                new_w = list(w)
                new_w[pos] = val
                idx = word_to_idx[tuple(new_w)]
                m |= (1 << idx)
        masks[i] = m
        
    all_mask = (1 << total_words) - 1
    
    # Greedy Construction
    uncovered = all_mask
    code_indices = []
    
    while uncovered:
        best_idx = -1
        best_count = -1
        # Find word covering most new words
        # Optimization: if time is tight, sample? No, 729 is small.
        for i in range(total_words):
            if i in code_indices: # Skip already chosen? 
                # Actually, we can just check all, but skipping chosen is slightly faster?
                # No, chosen words cover 0 new words (mostly), but might cover some if logic overlaps?
                # No, if i is in code, its mask is already in 'covered', so (mask[i] & uncovered) is 0.
                # So we can skip them.
                continue
            new_cov = masks[i] & uncovered
            cnt = new_cov.bit_count()
            if cnt > best_count:
                best_count = cnt
                best_idx = i
                if cnt == 0: break # No progress possible? Should not happen if uncovered != 0
        if best_idx == -1: break
        code_indices.append(best_idx)
        uncovered &= ~masks[best_idx]
        
    # Local Search: Removal
    # We maintain a list of indices.
    # Check time
    while time.time() < deadline - 5.0:
        improved = False
        # Try removing each codeword
        # To speed up, we can compute union of all others for each removal candidate?
        # That's O(N^2). N~75. 75^2 = 5625.
        # Or, we can just check if the codeword is "essential".
        # A codeword is essential if there is at least one word it covers that is NOT covered by any other codeword.
        
        # Efficient check for redundancy:
        # For each c in code:
        #   Let U_others = OR of masks of all c' in code where c' != c
        #   If (mask[c] & ~U_others) == 0, c is redundant.
        
        # We can precompute prefix/suffix sums of masks to do this in O(N) per pass?
        # Yes.
        N = len(code_indices)
        if N == 0: break
        
        prefix = [0] * (N + 1)
        for i in range(N):
            prefix[i+1] = prefix[i] | masks[code_indices[i]]
            
        suffix = [0] * (N + 1)
        for i in range(N-1, -1, -1):
            suffix[i] = suffix[i+1] | masks[code_indices[i]]
            
        for i in range(N):
            others_mask = prefix[i] | suffix[i+1]
            if (masks[code_indices[i]] & ~others_mask) == 0:
                # Redundant
                code_indices.pop(i)
                improved = True
                break # Restart pass as indices changed
                
        if not improved:
            # Try Swaps?
            # If removal didn't help, try to find a swap that might enable a removal later.
            # Or just accept the local minimum.
            # Given the budget, we can try a limited swap search.
            # Swap: remove c_i, add c_j.
            # Condition: (mask[c_j] | union_without_c_i) == all_mask
            # If we find such a swap, do it, and loop back to removal.
            
            # To avoid infinite loops, we can track the set of codewords or just rely on the fact that
            # swaps usually lead to a state where removal is possible.
            # However, simple swaps might cycle. 
            # Let's try one pass of swaps.
            swap_found = False
            for i in range(len(code_indices)):
                c_i = code_indices[i]
                others_mask = prefix[i] | suffix[i+1]
                # We need c_j such that mask[c_j] covers the "hole" left by c_i
                hole = masks[c_i] & ~others_mask
                if hole == 0:
                    continue # Should have been removed
                
                # Find any c_j not in code that covers 'hole'
                # hole is a subset of mask[c_i].
                # We can iterate through all words not in code.
                in_code_set = set(code_indices)
                for j in range(total_words):
                    if j in in_code_set: continue
                    if (masks[j] & hole) == hole:
                        # Valid swap
                        code_indices[i] = j
                        swap_found = True
                        break
                if swap_found:
                    break
            if not swap_found:
                break # Local minimum reached
                
    # Convert indices back to word lists
    return [words[i] for i in code_indices]
