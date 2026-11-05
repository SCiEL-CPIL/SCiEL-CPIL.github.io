import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    all_words = list(itertools.product(range(q), repeat=n))
    total_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbor indices
    neighbors_idx = []
    for w in all_words:
        neigh = set()
        neigh.add(w)
        for i in range(n):
            for val in range(q):
                if w[i] != val:
                    new_w = list(w)
                    new_w[i] = val
                    neigh.add(tuple(new_w))
        neighbors_idx.append([word_to_idx[x] for x in neigh])

    covered = [False] * total_words
    num_covered = 0
    codewords = []
    
    # Greedy
    while num_covered < total_words:
        best_idx = -1
        max_count = -1
        
        # To allow for some variation, we can shuffle the candidate list?
        # No, we need the max. Shuffling doesn't change the max.
        # But if we have a time limit, we might not find the global max if we stop early.
        # But 729 is small, so we always find the max.
        
        for i in range(total_words):
            count = 0
            for idx in neighbors_idx[i]:
                if not covered[idx]:
                    count += 1
            if count > max_count:
                max_count = count
                best_idx = i
                
        if best_idx == -1:
            break
            
        codewords.append(all_words[best_idx])
        
        for idx in neighbors_idx[best_idx]:
            if not covered[idx]:
                covered[idx] = True
                num_covered += 1
                
        if time.time() > deadline:
            break
            
    return [list(cw) for cw in codewords]
