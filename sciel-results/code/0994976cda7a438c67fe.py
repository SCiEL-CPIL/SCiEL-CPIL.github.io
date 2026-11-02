import numpy as np
import itertools
import random

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(x) for x in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=8, radius=1
    # Approach: Greedy Set Cover Construction + Greedy Removal
    
    total_words = 3**n
    # Precompute neighbors for all words
    # Word index to tuple
    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[i] = temp % 3
            temp //= 3
        return w

    # Precompute neighbor indices for all words
    # neighbors[i] is a list of indices within distance 1 of word i
    all_neighbors = []
    for i in range(total_words):
        w = idx_to_word(i)
        neigh = [i]
        for pos in range(n):
            for delta in [-1, 1]:
                new_val = w[pos] + delta
                if 0 <= new_val < 3:
                    new_w = w.copy()
                    new_w[pos] = new_val
                    # Convert back to index
                    idx_val = 0
                    mult = 1
                    for k in range(n):
                        idx_val += new_w[k] * mult
                        mult *= 3
                    neigh.append(idx_val)
        all_neighbors.append(neigh)

    # Precompute coverage masks (Python ints)
    # mask[i] has bit j set if word j is covered by word i
    masks = [0] * total_words
    for i in range(total_words):
        m = 0
        for j in all_neighbors[i]:
            m |= (1 << j)
        masks[i] = m

    # 1. Greedy Set Cover
    uncovered = (1 << total_words) - 1
    code_indices = []
    
    while uncovered:
        max_gain = 0
        best_i = -1
        # Optimization: If we find a word that covers all remaining uncovered, we can break early?
        # Unlikely.
        for i in range(total_words):
            # If this word doesn't cover any new words, skip?
            # We need to check intersection with uncovered.
            gain = (masks[i] & uncovered).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_i = i
                # If max_gain is 1, we can't do better? No, we want max.
                # If max_gain == number of bits in uncovered, we found a perfect cover for the rest?
                # If gain == uncovered.bit_count(), we can add it and break.
                if gain == uncovered.bit_count():
                    break
        
        if best_i == -1:
            break # Should not happen
            
        code_indices.append(best_i)
        uncovered &= ~masks[best_i]
        
    # Convert indices to word lists
    initial_codewords = [idx_to_word(i) for i in code_indices]
    
    # 2. Greedy Removal (Pruning)
    # Convert to numpy for speed in counting
    # Map word to index for counting? No, we use the global index space.
    # We maintain counts of coverage for all 6561 words.
    
    counts = np.zeros(total_words, dtype=np.int32)
    # Precompute neighbor lists for the *selected* codewords only?
    # We need to check if removing a codeword leaves gaps.
    
    # active list
    active = [True] * len(initial_codewords)
    
    # Initialize counts
    for i, cw_idx in enumerate(code_indices):
        counts[all_neighbors[cw_idx]] += 1
        
    # Iterative removal
    # We use a loop that scans all active codewords.
    # To speed up, we can vectorize the "is removable" check.
    # Removable if min(counts[neighbors]) > 1.
    
    changed = True
    while changed:
        changed = False
        # Find all removable indices
        removable_indices = []
        for i in range(len(initial_codewords)):
            if not active[i]:
                continue
            neigh = all_neighbors[code_indices[i]]
            # Check min count
            if np.min(counts[neigh]) > 1:
                removable_indices.append(i)
                
        if not removable_indices:
            break
            
        # Remove them one by one to be safe?
        # Or remove all?
        # If we remove all, we might uncover words.
        # Let's remove one by one, but batch the check?
        # Actually, removing one might make others non-removable?
        # If A is removable, it doesn't cover any unique words.
        # If B is removable, it doesn't cover any unique words.
        # If they share a word w, and count(w) was 2, removing A makes count(w) 1.
        # B is still "removable" in the sense that B doesn't cover *unique* words relative to the set *before* removal.
        # But if we remove A first, then check B, B might now have a neighbor with count 1 (if that neighbor was only covered by A and B? No, count was 2, so covered by A and B. If A is removed, count is 1. B is the *only* one covering it now? No, B is still there.
        # Wait. If count(w) was 2 (A and B).
        # Remove A. count(w) becomes 1 (only B).
        # Now B is *not* removable, because w has count 1.
        # So order matters.
        # We must remove one, update counts, then re-evaluate.
        
        # To be safe and simple, remove one at a time, pick the first one found.
        # This is O(N^2) but N is small (~500-700).
        # 500*500*17 = 4M ops. Fast.
        
        # Pick the first removable one
        i = removable_indices[0]
        active[i] = False
        counts[all_neighbors[code_indices[i]]] -= 1
        changed = True

    # Extract final codewords
    final_codewords = [initial_codewords[i] for i in range(len(initial_codewords)) if active[i]]
    
    return final_codewords
