import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # The task specifies q=3, n=8, radius=1.
    # We assume these are passed, but the logic is tailored for this specific hard case.
    # For generality, if q, n, radius differ, we fall back to the simple construction
    # or a generic greedy if feasible, but the prompt implies this specific task.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback to the simple construction for other parameters
        import itertools
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_words = q ** n
    # Pre-compute mapping from word to index
    # Index = w0 * 3^7 + w1 * 3^6 + ... + w7
    # We can generate all words and their indices
    
    # 1. Generate all words and their indices
    # words[i] is the word corresponding to index i
    # We need neighbors for each index i: all words j such that dist(words[i], words[j]) <= 1
    
    # Efficiently generate neighbors
    # For a given word w, neighbors are:
    # 1. w itself
    # 2. w with one position changed to one of the other 2 values
    
    # Let's pre-compute the neighbor indices for all 6561 words
    # This takes some memory: 6561 * 17 ints. ~110k ints. Trivial.
    
    neighbors = np.zeros((total_words, 1 + n * (q - 1)), dtype=np.int32)
    
    # Helper to convert index to word
    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            w[i] = temp % q
            temp //= q
        return w
        
    # Helper to convert word to index
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    # Pre-compute neighbors
    for i in range(total_words):
        w = idx_to_word(i)
        neigh_list = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                neigh_list.append(word_to_idx(w))
            w[pos] = orig_val
        neighbors[i] = neigh_list

    # 2. Greedy Set Cover
    uncovered = np.ones(total_words, dtype=bool)
    selected_indices = []
    
    # To speed up argmax, we can just compute counts for all candidates each time.
    # counts[i] = number of uncovered words in neighbors[i]
    
    # Optimization: 
    # Instead of recomputing `uncovered[neighbors]` fully (which is a large fancy index),
    # we can just do it. It's 6561 * 17 elements. 
    # NumPy handles this very fast.
    
    while True:
        if time.time() > deadline:
            break
            
        # Calculate coverage of each candidate among currently uncovered words
        # uncovered[neighbors] creates a boolean array of shape (6561, 17)
        # .sum(axis=1) gives an array of shape (6561,)
        counts = uncovered[neighbors].sum(axis=1)
        
        best_idx = np.argmax(counts)
        best_count = counts[best_idx]
        
        if best_count == 0:
            # No word can cover any uncovered word? This implies we missed some words
            # or the code is not valid. In a valid covering problem, this shouldn't happen
            # unless we ran out of candidates, but we have all candidates.
            # If this happens, it means the remaining uncovered words are isolated?
            # No, every word is a neighbor of itself. So if any word is uncovered,
            # its own count must be >= 1.
            # So best_count == 0 implies all words are covered.
            break
            
        # Select this codeword
        selected_indices.append(best_idx)
        
        # Update uncovered set
        uncovered[neighbors[best_idx]] = False
        
        # Check if all covered
        if not uncovered.any():
            break

    # Convert selected indices back to words
    codewords = [idx_to_word(i) for i in selected_indices]
    
    return codewords
