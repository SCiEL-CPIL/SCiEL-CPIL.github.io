import numpy as np
import itertools
import time

def construct(q, n, radius):
    # The task is specifically for q=3, n=6, radius=1
    # But the function signature is generic.
    # If q, n, radius are not 3, 6, 1, we fall back to a simple construction or error.
    # Given the context, it will be called with 3, 6, 1.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback to the seed logic or a generic greedy if possible
        # For now, return the seed result for non-target cases to be safe
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 4:
                return tetracode
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Small n, just return all words?
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Total words
    total_words = 3**6
    words = np.array(list(itertools.product(range(3), repeat=6)), dtype=np.int8)
    
    # Precompute neighbors for each word index
    # neighbor[i] is a list of indices j such that dist(words[i], words[j]) <= 1
    neighbors = []
    for i in range(total_words):
        w = words[i]
        neigh = [i]
        for pos in range(6):
            orig = w[pos]
            for val in range(3):
                if val == orig:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                # Find index of new_w
                # Since we have the list, we can map tuple to index
                # But doing this in loop is slow. 
                # Better: precompute a dictionary from tuple to index.
                pass
        
    # Let's use a dictionary for fast lookup
    word_to_idx = {tuple(w): i for i, w in enumerate(words)}
    
    neighbors = []
    for i in range(total_words):
        w = words[i]
        neigh_set = {i}
        for pos in range(6):
            orig = w[pos]
            for val in range(3):
                if val == orig:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                neigh_set.add(word_to_idx[tuple(new_w)])
        neighbors.append(list(neigh_set))
        
    covered = np.zeros(total_words, dtype=bool)
    code_indices = []
    
    while not covered.all():
        if time.time() > deadline:
            break
            
        best_idx = -1
        best_count = -1
        
        # To speed up, we can skip indices that are already covered? 
        # No, a covered word can still cover other uncovered words.
        # But if a word is covered, it's in the "ball" of some existing codeword.
        # Does it help to pick a covered word as a new codeword?
        # Yes, it can cover other uncovered words.
        
        for i in range(total_words):
            # Optional optimization: if i is covered and we have many uncovered, 
            # maybe prefer uncovered? 
            # Let's just count.
            count = 0
            for j in neighbors[i]:
                if not covered[j]:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = i
                
        if best_idx == -1:
            break
            
        code_indices.append(best_idx)
        for j in neighbors[best_idx]:
            covered[j] = True
            
    # Convert indices back to lists of integers
    code = [words[i].tolist() for i in code_indices]
    
    # Verify validity just in case
    # (Omitted for speed in final output, but good for dev)
    
    return code
