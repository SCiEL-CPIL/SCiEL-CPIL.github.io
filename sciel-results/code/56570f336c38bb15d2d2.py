import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    # Parameters specific to the task, but function signature is generic.
    # We assume q=3, n=8, radius=1 for the optimization logic, 
    # but we should handle general inputs gracefully or fall back to seed if not 3,8,1.
    if q != 3 or n != 8 or radius != 1:
        # Fallback to the seed logic for other parameters
        # The seed logic was: direct sum of tetracode
        # This only works for specific n. For general case, a simple greedy is safer.
        # But the task is specifically for 8 matches, ternary.
        # Let's implement a general greedy for safety, but optimize for 3,8,1.
        pass

    # Precomputation for n=8, q=3
    N = q ** n
    # Generate all words as tuples of ints
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Precompute neighbors for each word index
    # neighbors[i] contains indices j such that dist(all_words[i], all_words[j]) <= radius
    # For radius 1, dist is 0 or 1.
    neighbors = []
    for i, w in enumerate(all_words):
        neigh = [i] # distance 0
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                # Create new word
                new_w = list(w)
                new_w[pos] = val
                # Find index of new_w
                # To make this fast, we can use a dictionary mapping tuple to index
                # But doing this inside the loop is slow.
                pass
        
    # Better precomputation strategy:
    # Create a dict mapping tuple to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    neighbors = []
    for i, w in enumerate(all_words):
        neigh_set = set()
        neigh_set.add(i)
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh_set.add(word_to_idx[new_w])
        neighbors.append(list(neigh_set))

    # Greedy Set Cover
    uncovered = np.ones(N, dtype=bool)
    C = []
    
    # Optimization: Keep track of uncovered count for each candidate
    # initially, every candidate covers 17 words.
    # We just scan all candidates each time.
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s for reduction and output
    
    while np.any(uncovered):
        best_idx = -1
        best_count = -1
        
        # Iterate through all possible codewords
        # We can skip indices that are already in C? 
        # Actually, adding a duplicate is useless. 
        # But checking if i is in C is O(len(C)). 
        # Better: just pick the one with max new coverage. If it's in C, count will be 0 (since uncovered is updated).
        
        for i in range(N):
            # Count how many neighbors of i are uncovered
            # neighbors[i] is a list of indices
            cnt = np.sum(uncovered[neighbors[i]])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
                
        if best_count == 0:
            # Should not happen if uncovered is not empty, unless there's a bug
            break
            
        C.append(best_idx)
        # Update uncovered
        uncovered[neighbors[best_idx]] = False
        
        # Time check
        if time.time() - start_time > time_limit:
            break

    # Reduction Phase
    # We need to remove redundant codewords.
    # A codeword c is redundant if all words in neighbors[c] are covered by other codewords in C.
    # This is equivalent to: for all w in neighbors[c], the number of codewords in C that cover w is >= 2.
    
    # Initialize coverage counts
    coverage_counts = np.zeros(N, dtype=int)
    for c in C:
        coverage_counts[neighbors[c]] += 1
        
    changed = True
    while changed:
        changed = False
        new_C = []
        for c in C:
            # Check if c is redundant
            # min coverage in neighbors[c] must be >= 2
            if np.min(coverage_counts[neighbors[c]]) >= 2:
                # Redundant, remove it
                coverage_counts[neighbors[c]] -= 1
                changed = True
            else:
                new_C.append(c)
        C = new_C

    # Convert indices back to lists of ints
    result = [list(all_words[i]) for i in C]
    return result
