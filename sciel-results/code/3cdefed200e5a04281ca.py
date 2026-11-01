import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # 1. Precompute all words and their neighbors
    # Correct way to generate ternary words
    words = []
    for i in range(3**n):
        w = []
        v = i
        for _ in range(n):
            w.append(v % 3)
            v //= 3
        words.append(w)
    words = np.array(words)
    
    # Map word tuple to index
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    # Precompute coverage for each word
    # cover[i] is a boolean array of length 3**n indicating which words are within radius
    num_words = 3**n
    covers = np.zeros((num_words, num_words), dtype=bool)
    for i in range(num_words):
        w = words[i]
        for j in range(num_words):
            # Calculate Hamming distance
            # Optimization: use numpy
            dist = np.sum(w != words[j])
            if dist <= radius:
                covers[i, j] = True
                
    # This O(N^2 * n) is 729^2 * 6 ~ 3 million ops, very fast.
    # But we can do better with precomputed neighbors.
    
    # Let's re-do coverage generation more efficiently
    covers = []
    for i in range(num_words):
        w = words[i]
        mask = np.zeros(num_words, dtype=bool)
        # Self
        mask[i] = True
        # Neighbors at distance 1
        for pos in range(n):
            for val in range(q):
                if val == w[pos]: continue
                new_w = w.copy()
                new_w[pos] = val
                idx = word_to_idx[tuple(new_w)]
                mask[idx] = True
        covers.append(mask)
    covers = np.array(covers)
    
    # 2. Initialize with direct sum construction
    # Tetracode for n=4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in range(3**(n-4)):
            # Generate tail
            t = []
            v = tail
            for _ in range(n-4):
                t.append(v % 3)
                v //= 3
            initial_codewords.append(h + t)
            
    # Convert to indices
    current_codewords = [word_to_idx[tuple(c)] for c in initial_codewords]
    
    # 3. Local Search
    # State: set of codeword indices
    # We track coverage counts
    counts = np.zeros(num_words, dtype=int)
    for c in current_codewords:
        counts += covers[c]
        
    best_size = len(current_codewords)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to remove redundant codewords
        for i in range(len(current_codewords)):
            c = current_codewords[i]
            # Check if any word covered by c has count == 1
            # If all words covered by c have count > 1, c is redundant
            is_redundant = True
            for j in np.where(covers[c])[0]:
                if counts[j] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove c
                counts -= covers[c]
                current_codewords.pop(i)
                best_size = len(current_codewords)
                improved = True
                break # Restart scan from beginning
        
        if improved:
            continue
            
        # If no removal, try swaps
        # For each codeword c_out, find c_in that covers the orphans of c_out
        # Orphans: words covered by c_out with count == 1
        swap_done = False
        for i, c_out in enumerate(current_codewords):
            # Identify orphans
            orphans = []
            for j in np.where(covers[c_out])[0]:
                if counts[j] == 1:
                    orphans.append(j)
            
            if not orphans:
                # This codeword is redundant? 
                # Wait, if orphans is empty, it means all words it covers have count > 1.
                # Then it should have been removed in the previous step.
                # So this case shouldn't happen if the removal loop was thorough.
                # But maybe the removal loop missed it?
                # Let's just try to swap it with something to see if it helps?
                # Actually, if orphans is empty, it IS redundant.
                # Why wasn't it removed?
                # Maybe I broke out of the loop too early?
                # The loop `for i in range...` breaks on first removal.
                # So if I found a redundant one, I removed it and `improved=True`, `break`.
                # So if we are here, NO codeword is redundant.
                # Thus, for every c_out, there is at least one orphan.
                pass
            
            # We need a c_in that covers ALL orphans
            # c_in cannot be in current_codewords
            # And c_in should ideally not be c_out (it can't be, c_out is in set)
            
            # Optimization: Precompute for each set of orphans? 
            # Or just iterate all 729 candidates.
            # 729 * 73 * 13 (orphan check) is small.
            
            for c_in in range(num_words):
                if c_in in current_codewords:
                    continue
                    
                # Check if c_in covers all orphans
                valid = True
                for o in orphans:
                    if not covers[c_in, o]:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Try the swap
                # Remove c_out
                counts -= covers[c_out]
                current_codewords[i] = c_in
                counts += covers[c_in]
                
                # Check if this swap allows a removal
                # We need to check if ANY codeword is now redundant
                # Note: c_in might be redundant, or some other codeword.
                # Also, c_out was removed, so its orphans are now covered by c_in.
                # Other codewords might have become redundant because their unique coverage is now shared?
                # No, adding a codeword increases counts, so it can't make a codeword redundant 
                # unless the codeword we removed was the *only* one covering some points 
                # that are now covered by c_in, and those points were *also* covered by 
                # another codeword which is now redundant?
                # Actually, removing c_out decreases counts. Adding c_in increases counts.
                # Net effect on counts is complex.
                
                # Let's just check for redundancy in the new set.
                removed_after_swap = False
                for k in range(len(current_codewords)):
                    c_k = current_codewords[k]
                    is_red = True
                    for j in np.where(covers[c_k])[0]:
                        if counts[j] == 1:
                            is_red = False
                            break
                    if is_red:
                        # Remove it
                        counts -= covers[c_k]
                        current_codewords.pop(k)
                        best_size = len(current_codewords)
                        improved = True
                        removed_after_swap = True
                        break
                        
                if removed_after_swap:
                    swap_done = True
                    break
                else:
                    # Revert swap
                    counts -= covers[c_in]
                    current_codewords[i] = c_out
                    counts += covers[c_out]
                    
            if swap_done:
                break
                
    return [list(words[c]) for c in current_codewords]
