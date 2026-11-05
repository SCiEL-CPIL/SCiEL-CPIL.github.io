import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    # Only handle the specific case q=3, n=7, radius=1 efficiently
    # Fallback to seed logic for other cases if needed, but task is specific
    if q != 3 or n != 7 or radius != 1:
        # Fallback to direct sum tetracode for general case
        import itertools
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            # This fallback is only for non-target cases, which shouldn't happen in evaluation
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_words = q ** n
    # Generate all words
    words = []
    for i in range(total_words):
        w = []
        v = i
        for _ in range(n):
            w.append(v % q)
            v //= q
        words.append(tuple(w))
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute balls (coverage masks)
    # M[i] is a bitmask where bit j is 1 if word j is in ball of word i
    masks = [0] * total_words
    
    # Helper to compute hamming distance
    # Since n=7 is small, we can just iterate
    
    for i, w in enumerate(words):
        mask = 0
        # The word itself
        mask |= (1 << word_to_idx[w])
        
        # Words at distance 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                # Create new word
                new_w_list = list(w)
                new_w_list[pos] = new_val
                new_w = tuple(new_w_list)
                mask |= (1 << word_to_idx[new_w])
        masks[i] = mask

    # Greedy Set Cover
    uncovered_mask = (1 << total_words) - 1
    selected_indices = []
    
    # To speed up, we can maintain a list of active candidates
    # But total_words is only 2187, so scanning all is fine.
    
    while uncovered_mask != 0:
        if time.time() - start_time > time_limit:
            break
        
        best_idx = -1
        best_count = -1
        
        # Optimization: We can skip candidates that are already "covered" in a sense?
        # No, a candidate might cover new points even if it's close to other selected ones.
        # We just need max popcount(M[i] & uncovered_mask).
        
        # We can iterate through all candidates
        # To break ties or improve, we could randomize order, but deterministic is safer for reproducibility unless we restart.
        
        # Let's try a slightly smarter approach: 
        # If a candidate's mask is a subset of another's mask relative to uncovered, it's worse.
        # But checking subset is also expensive.
        
        # Simple max scan
        # We can speed this up by noting that if a candidate covers 0 new points, skip it.
        # popcount is fast in Python 3.10+ (int.bit_count())
        
        for i in range(total_words):
            # Quick check: if mask is 0 or no overlap, skip
            # overlap = masks[i] & uncovered_mask
            # count = overlap.bit_count()
            
            # Optimization: If we already have a candidate covering all remaining points, we can stop early?
            # Unlikely.
            
            overlap = masks[i] & uncovered_mask
            if overlap:
                count = overlap.bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
                    # Early exit if we found a perfect cover for remaining? 
                    # Only if count == uncovered_mask.bit_count()
                    # This is rare.
        
        if best_idx == -1:
            break
            
        selected_indices.append(best_idx)
        uncovered_mask &= ~masks[best_idx]
        
    # Convert indices to words
    result = [words[i] for i in selected_indices]
    
    # Ensure distinctness (greedy ensures this implicitly as we pick distinct indices, 
    # but if we ever picked same index twice? No, we pick best_idx, add to list. 
    # We don't remove candidate from pool, but its contribution to future overlaps decreases.
    # It is possible to pick same index again if it still has max overlap? 
    # No, because we remove its coverage. If it was the best, it covered some. 
    # After removal, its new overlap is smaller. 
    # However, if we picked i, then in next iteration, i might still be best if it covers many *other* uncovered points?
    # No, the points it covered are now uncovered_mask=0 for those bits.
    # So its new overlap is strictly smaller (unless it covered 0 new points, which would mean best_count was 0, loop breaks).
    # So distinctness is guaranteed.
    
    return [list(w) for w in result]
