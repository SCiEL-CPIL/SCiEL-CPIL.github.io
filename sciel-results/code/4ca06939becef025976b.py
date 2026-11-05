"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np


def construct(q, n, radius):
    # Fallback for other parameters
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            all_words = list(itertools.product(range(q), repeat=n))
            return all_words

    # Specific optimization for q=3, n=8, radius=1
    # Start with the direct sum of the ternary tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
    
    m_initial = len(code)
    if m_initial == 0:
        return code

    # Helper to convert word to base-3 index
    def word_to_idx(word):
        idx = 0
        for val in word:
            idx = idx * 3 + val
        return idx

    # Precompute neighbors for each codeword
    # neighbors[i] is a list of 17 indices
    neighbors = []
    for c in code:
        c_idx = word_to_idx(c)
        neigh = [c_idx]
        for pos in range(n):
            orig = c[pos]
            for val in range(3):
                if val != orig:
                    # Create modified word
                    # Optimization: instead of copying list, we can compute index change
                    # idx = sum(w_i * 3^(n-1-i))
                    # Change at pos i: delta = (val - orig) * 3^(n-1-i)
                    # But we need the new word to compute idx? No, we can just compute new idx directly.
                    # Let's stick to simple list manipulation for clarity first, then optimize if needed.
                    # Given n=8, 486 words, 17 neighbors, this is small.
                    new_c = c[:]
                    new_c[pos] = val
                    neigh.append(word_to_idx(new_c))
        neighbors.append(neigh)

    # Coverage array
    coverage = np.zeros(6561, dtype=np.int32)
    for neigh in neighbors:
        coverage[neigh] += 1

    # Peeling loop
    # We maintain a list of active codeword indices
    active = list(range(m_initial))
    # To allow removal, we map active indices to their original positions or just keep the data.
    # It's easier to keep `code`, `neighbors` as lists and pop/remove.
    # But `neighbors` contains indices into the global space, so they don't change.
    # `code` list needs to be updated.
    
    # We will repeatedly find the best removable codeword.
    # To speed up, we can precompute the "score" for each codeword.
    # Score = sum(coverage[neigh]) for neigh in neighbors[i]
    # But coverage changes.
    
    # Heuristic: Remove codeword if min(coverage[neighbors[i]]) >= 2.
    # Among those, pick one with max sum(coverage[neighbors[i]]).
    
    import random
    # random.seed(42) # For reproducibility if needed, but let's try random restarts if stuck?
    # The problem asks for a single run.
    
    while True:
        best_idx = -1
        best_score = 0
        
        for i in active:
            neigh = neighbors[i]
            # Check removability
            # If any neighbor has coverage 1, it's not removable
            # We can use numpy min for speed
            min_cov = np.min(coverage[neigh])
            if min_cov >= 2:
                # Calculate score
                score = np.sum(coverage[neigh])
                if score > best_score:
                    best_score = score
                    best_idx = i

        if best_idx == -1:
            break
        
        # Remove best_idx
        # Update coverage
        coverage[neighbors[best_idx]] -= 1
        
        # Remove from active list
        # We need to remove the element at position corresponding to best_idx in the 'active' list?
        # No, 'active' contains the original indices.
        # We just remove best_idx from the 'active' list.
        active.remove(best_idx)
        
        # We don't strictly need to remove from 'code' or 'neighbors' unless we return them.
        # We return the code words for indices in 'active'.
        
    # Construct result
    result = [code[i] for i in active]
    return result
# EVOLVE-BLOCK-END
