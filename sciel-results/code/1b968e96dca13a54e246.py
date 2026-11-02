import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = 115.0 # 5s margin for 120s limit
    
    total_words = q ** n
    
    # If the problem is small enough, we do exact/heuristic search.
    # For 3^7 = 2187, it's small.
    
    # Generate all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i) # self
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                new_idx = word_to_idx[tuple(new_w)]
                neighbors[i].append(new_idx)
                
    # Convert to numpy arrays for fast indexing
    neighbor_arrays = [np.array(neigh, dtype=int) for neigh in neighbors]
    
    # Greedy Set Cover
    uncovered = np.ones(total_words, dtype=bool)
    selected = []
    
    while np.any(uncovered):
        if time.time() > deadline:
            break
        
        best_idx = -1
        max_covered = 0
        
        for c in range(total_words):
            # Count uncovered in neighbors of c
            cnt = np.sum(uncovered[neighbor_arrays[c]])
            if cnt > max_covered:
                max_covered = cnt
                best_idx = c
        
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        uncovered[neighbor_arrays[best_idx]] = False
        
    # Ensure validity if time ran out
    if np.any(uncovered):
        # Quick fix: add words to cover remaining
        while np.any(uncovered) and time.time() < deadline:
            uncov_idx = np.where(uncovered)[0][0]
            # Add the word itself
            c = uncov_idx
            selected.append(c)
            uncovered[neighbor_arrays[c]] = False
            
    # Reduction Phase
    counts = np.zeros(total_words, dtype=int)
    for c in selected:
        counts[neighbor_arrays[c]] += 1
        
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate in reverse to handle pop
        for i in range(len(selected) - 1, -1, -1):
            if time.time() > deadline:
                break
            c = selected[i]
            min_count = np.min(counts[neighbor_arrays[c]])
            if min_count >= 2:
                counts[neighbor_arrays[c]] -= 1
                selected.pop(i)
                improved = True
                
    result = [all_words[idx] for idx in selected]
    return result
