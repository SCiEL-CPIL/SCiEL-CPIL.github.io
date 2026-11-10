import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    limit = 115.0 # Safety margin
    
    total_words = q ** n
    
    # Precompute neighbors for all words
    # neighbors[idx] = list of indices
    # neigh_bits[idx] = bitset of neighbors
    
    # Efficient neighbor generation
    # For each index, generate the word, then modify each position
    # Since q=3, n=6, this is small.
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    # Precompute base powers
    powers = [q ** (n - 1 - i) for i in range(n)]
    
    neigh_bits = [0] * total_words
    for idx in range(total_words):
        w = idx_to_word(idx)
        bits = 1 << idx
        for i in range(n):
            orig = w[i]
            # Try all other values
            for new_val in range(q):
                if new_val == orig:
                    continue
                w[i] = new_val
                n_idx = word_to_idx(w)
                bits |= (1 << n_idx)
            w[i] = orig
        neigh_bits[idx] = bits

    full_mask = (1 << total_words) - 1
    best_code = []
    best_size = total_words # Upper bound
    
    while time.time() - start_time < limit:
        # Run one instance of Greedy + Polish
        code = []
        uncovered_bits = full_mask
        covered_bits = 0
        
        # Greedy Phase
        while uncovered_bits:
            if time.time() - start_time > limit - 1.0: # Leave time for polish and return
                break
            
            best_idx = -1
            best_count = -1
            best_count_val = -1
            
            # Find candidate with max new coverage
            # To allow random tie-breaking, we collect all max candidates
            max_candidates = []
            
            # Optimization: 
            # We iterate through all possible centers.
            # For each, count = (neigh_bits[c] & uncovered_bits).bit_count()
            
            for c in range(total_words):
                count = (neigh_bits[c] & uncovered_bits).bit_count()
                if count > best_count:
                    best_count = count
                    max_candidates = [c]
                elif count == best_count and count > 0:
                    max_candidates.append(c)
            
            if best_count == 0:
                break
            
            # Random tie-breaking
            best_idx = random.choice(max_candidates)
            
            code.append(best_idx)
            covered_bits |= neigh_bits[best_idx]
            uncovered_bits &= ~neigh_bits[best_idx]
            
        # Polish Phase: Redundancy Removal
        # Iterate until no more removals
        changed = True
        while changed:
            changed = False
            if time.time() - start_time > limit - 0.5:
                break
                
            # Calculate union of all neighbors
            # We can do this by OR-ing all bits in code
            union_bits = 0
            for c in code:
                union_bits |= neigh_bits[c]
                
            # Check each codeword for redundancy
            new_code = []
            for c in code:
                # If removing c leaves union unchanged, it's redundant
                # union_others = union_bits with c's bits removed? No, OR is not invertible.
                # We must recompute union without c.
                # To speed up, we can precompute prefix/suffix ORs or just do O(N) per check.
                # N is small (~80), so O(N^2) total is fine.
                
                union_others = 0
                for c2 in code:
                    if c2 != c:
                        union_others |= neigh_bits[c2]
                
                # If neigh_bits[c] is a subset of union_others, c is redundant
                if (neigh_bits[c] & ~union_others) == 0:
                    changed = True
                    # Do not add c to new_code
                else:
                    new_code.append(c)
            
            code = new_code
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
    # Convert best_code indices to words
    result = [idx_to_word(idx) for idx in best_code]
    
    # Ensure distinctness (already guaranteed by greedy picking distinct centers? 
    # Yes, if a center is picked, its neighbors are covered. 
    # If we pick the same center again, count would be 0 (since it's covered). 
    # So centers are distinct.)
    
    return result
